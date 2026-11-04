package bedrock

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Fork-only (everc) Nova embedding behaviour, kept apart from upstream's Nova converter so a
// rebase onto a new upstream release touches only the hook in BedrockProvider.Embedding.
//
// Upstream's Nova path (ToBedrockNovaEmbeddingRequest) embeds exactly one modality per
// request, takes media as typed content parts and has no pooling. Our callers use the older
// contract, which this file keeps:
//
//   - text, and images in the "inputImage" / "inputImages" extra params (data URI, http(s)
//     URL or raw base64), in one request; text that is only whitespace counts as no text (the
//     HTTP entry points send a single space for an image-only request, see
//     transports/bifrost-http/lib/embedding_legacy.go);
//   - a text+image ("hybrid") request is fanned out into one text call plus one call per
//     valid image, and the vectors are mean-pooled and L2-normalised into one embedding;
//   - invalid or oversized images are skipped when something else is left to embed;
//   - "input_type" (Cohere-style hint) maps to a Nova embeddingPurpose; an explicit
//     "embeddingPurpose" extra param wins, then the request's task_type.
//
// A request that carries typed media parts takes upstream's path instead (see
// useLegacyNovaEmbedding), so both contracts work and neither is silently reinterpreted.

const (
	// Legacy extra params carrying images: one string, or a list of strings.
	legacyEmbeddingExtraParamInputImage  = "inputImage"
	legacyEmbeddingExtraParamInputImages = "inputImages"

	legacyNovaExtraParamInputType = "input_type"

	// legacyNovaMaxBase64ImageLen: Nova accepts about 5 MB of raw image, which is about 6.7 MB
	// base64-encoded. Anything above ~6.5 MB encoded is refused as "image too large" rather
	// than forwarded as a doomed request.
	legacyNovaMaxBase64ImageLen = 6_500_000

	legacyNovaDefaultPurpose = "GENERIC_INDEX"
)

// legacyEmbeddingImages returns the images carried by the legacy extra params, in order:
// "inputImage" first, then every entry of "inputImages". Blank strings and non-string entries
// are skipped. Both []string and []interface{} are accepted, because extra params usually
// arrive through a JSON round trip.
func legacyEmbeddingImages(params *schemas.EmbeddingParameters) []string {
	if params == nil || params.ExtraParams == nil {
		return nil
	}
	var images []string
	if s, ok := params.ExtraParams[legacyEmbeddingExtraParamInputImage].(string); ok && strings.TrimSpace(s) != "" {
		images = append(images, s)
	}
	switch v := params.ExtraParams[legacyEmbeddingExtraParamInputImages].(type) {
	case []string:
		for _, s := range v {
			if strings.TrimSpace(s) != "" {
				images = append(images, s)
			}
		}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				images = append(images, s)
			}
		}
	}
	return images
}

// useLegacyNovaEmbedding reports whether a Nova embedding request follows our legacy
// contract: no typed media part anywhere in the input. Text-only requests, and text or
// nothing plus "inputImage"/"inputImages" extra params, qualify. A request with an image,
// audio, video or file part goes to upstream's converter unchanged.
func useLegacyNovaEmbedding(request *schemas.BifrostEmbeddingRequest) bool {
	if request == nil {
		return false
	}
	for _, item := range request.Input {
		for _, part := range item.Content {
			switch part.Type {
			case schemas.EmbeddingContentPartTypeImage,
				schemas.EmbeddingContentPartTypeAudio,
				schemas.EmbeddingContentPartTypeVideo,
				schemas.EmbeddingContentPartTypeFile:
				return false
			}
		}
	}
	return true
}

// legacyNovaPurpose maps a Cohere-style input_type hint to a Nova embeddingPurpose.
// Index-time inputs map to GENERIC_INDEX and query-time inputs to GENERIC_RETRIEVAL;
// unknown or empty values fall back to GENERIC_INDEX.
func legacyNovaPurpose(inputType string) string {
	switch strings.ToLower(strings.TrimSpace(inputType)) {
	case "search_query", "query", "retrieval", "generic_retrieval":
		return "GENERIC_RETRIEVAL"
	case "search_document", "document", "index", "generic_index":
		return "GENERIC_INDEX"
	case "classification":
		return "CLASSIFICATION"
	case "clustering":
		return "CLUSTERING"
	case "image", "image_retrieval":
		return "IMAGE_RETRIEVAL"
	default:
		return legacyNovaDefaultPurpose
	}
}

// resolveLegacyNovaPurpose picks the embeddingPurpose for a request: an explicit
// embeddingPurpose extra param wins (raw enum passthrough), then the input_type hint, then
// the request's task_type (upstream's field), then the default.
func resolveLegacyNovaPurpose(extra map[string]interface{}, taskType *string) string {
	if s, ok := extra[BedrockNovaExtraParamEmbeddingPurpose].(string); ok && strings.TrimSpace(s) != "" {
		return s
	}
	if s, ok := extra[legacyNovaExtraParamInputType].(string); ok && strings.TrimSpace(s) != "" {
		return legacyNovaPurpose(s)
	}
	if taskType != nil && strings.TrimSpace(*taskType) != "" {
		return *taskType
	}
	return legacyNovaDefaultPurpose
}

// cleanLegacyNovaBase64 strips whitespace (newlines, carriage returns, spaces, tabs) from a
// base64 payload.
func cleanLegacyNovaBase64(b64 string) string {
	replacer := strings.NewReplacer("\n", "", "\r", "", " ", "", "\t", "")
	return strings.TrimSpace(replacer.Replace(b64))
}

// legacyEmbeddingText joins the text parts of every input item the way the older API joined
// a text list: a single text is used as is, several texts are each followed by " \n".
func legacyEmbeddingText(input []schemas.EmbeddingInputItem) string {
	var texts []string
	for _, item := range input {
		for _, part := range item.Content {
			if part.Type == schemas.EmbeddingContentPartTypeText && part.Text != nil {
				texts = append(texts, *part.Text)
			}
		}
	}
	switch len(texts) {
	case 0:
		return ""
	case 1:
		return texts[0]
	}
	var b strings.Builder
	for _, t := range texts {
		b.WriteString(t)
		b.WriteString(" \n")
	}
	return b.String()
}

// resolveLegacyNovaImage normalises one image input (data URI, URL or raw base64) into Nova's
// format plus source, applying base64 cleaning and the encoded-size cap. It returns an error
// for an invalid or oversized image so the caller can skip it (hybrid) or fail (image-only).
func resolveLegacyNovaImage(ctx context.Context, model, image string, detailLevel *BedrockNovaDetailLevel) (*BedrockNovaEmbeddingImage, error) {
	src, err := convertImageToBedrockSource(ctx, model, image)
	if err != nil {
		return nil, fmt.Errorf("invalid image input: %w", err)
	}
	if src == nil {
		return nil, fmt.Errorf("invalid image input: no image source")
	}
	if src.Source.Bytes == nil {
		if src.Source.S3Location != nil {
			return &BedrockNovaEmbeddingImage{
				Format:      src.Format,
				Source:      BedrockNovaEmbeddingSource{S3Location: src.Source.S3Location},
				DetailLevel: detailLevel,
			}, nil
		}
		return nil, fmt.Errorf("invalid image input: no image bytes")
	}
	clean := cleanLegacyNovaBase64(*src.Source.Bytes)
	if clean == "" {
		return nil, fmt.Errorf("invalid image input: empty image bytes")
	}
	if len(clean) > legacyNovaMaxBase64ImageLen {
		return nil, fmt.Errorf("image too large: %d base64 bytes exceeds limit of %d", len(clean), legacyNovaMaxBase64ImageLen)
	}
	return &BedrockNovaEmbeddingImage{
		Format:      src.Format,
		Source:      BedrockNovaEmbeddingSource{Bytes: &clean},
		DetailLevel: detailLevel,
	}, nil
}

// buildLegacyNovaEmbeddingRequests builds the ordered list of Nova single-embedding requests
// for a legacy-contract request: one text request when text is present, then one request per
// valid image. It returns an error when nothing embeddable remains, surfacing the last image
// error (such as "image too large") so the caller gets an actionable message.
func buildLegacyNovaEmbeddingRequests(ctx context.Context, bifrostReq *schemas.BifrostEmbeddingRequest) ([]*BedrockNovaEmbeddingRequest, error) {
	if bifrostReq == nil {
		return nil, fmt.Errorf("bifrost embedding request is nil")
	}

	var extra map[string]interface{}
	var dimensions *int
	var taskType *string
	if bifrostReq.Params != nil {
		extra = bifrostReq.Params.ExtraParams
		dimensions = bifrostReq.Params.Dimensions
		taskType = bifrostReq.Params.TaskType
	}
	purpose := BedrockNovaEmbeddingPurpose(resolveLegacyNovaPurpose(extra, taskType))

	truncationMode := BedrockNovaTruncationModeEnd
	if s, ok := extra[BedrockNovaExtraParamTruncationMode].(string); ok && strings.TrimSpace(s) != "" {
		truncationMode = BedrockNovaTruncationMode(s)
	}
	var detailLevel *BedrockNovaDetailLevel
	if s, ok := extra[BedrockNovaExtraParamDetailLevel].(string); ok && strings.TrimSpace(s) != "" {
		d := BedrockNovaDetailLevel(s)
		detailLevel = &d
	}

	var requests []*BedrockNovaEmbeddingRequest
	if text := legacyEmbeddingText(bifrostReq.Input); strings.TrimSpace(text) != "" {
		value := text
		requests = append(requests, &BedrockNovaEmbeddingRequest{
			TaskType: TaskTypeSingleEmbedding,
			SingleEmbeddingParams: &BedrockNovaSingleEmbeddingParams{
				EmbeddingPurpose:   purpose,
				EmbeddingDimension: dimensions,
				Text:               &BedrockNovaEmbeddingText{TruncationMode: truncationMode, Value: &value},
			},
		})
	}

	var lastImageErr error
	for _, image := range legacyEmbeddingImages(bifrostReq.Params) {
		img, err := resolveLegacyNovaImage(ctx, bifrostReq.Model, image, detailLevel)
		if err != nil {
			// Skip an invalid or oversized image; remember why in case nothing else is left.
			lastImageErr = err
			continue
		}
		requests = append(requests, &BedrockNovaEmbeddingRequest{
			TaskType: TaskTypeSingleEmbedding,
			SingleEmbeddingParams: &BedrockNovaSingleEmbeddingParams{
				EmbeddingPurpose:   purpose,
				EmbeddingDimension: dimensions,
				Image:              img,
			},
		})
	}

	if len(requests) == 0 {
		if lastImageErr != nil {
			return nil, lastImageErr
		}
		return nil, fmt.Errorf("no input provided for embedding")
	}
	return requests, nil
}

// legacyMeanPoolAndNormalize element-wise mean-pools the vectors and L2-normalises the result.
// It returns nil for no vectors and the single vector unchanged for exactly one. Vectors
// shorter than the longest are zero-padded, so an odd response cannot panic.
func legacyMeanPoolAndNormalize(vecs [][]float64) []float64 {
	if len(vecs) == 0 {
		return nil
	}
	if len(vecs) == 1 {
		return vecs[0]
	}

	dim := 0
	for _, v := range vecs {
		if len(v) > dim {
			dim = len(v)
		}
	}
	if dim == 0 {
		return []float64{}
	}

	pooled := make([]float64, dim)
	for _, v := range vecs {
		for i, x := range v {
			pooled[i] += x
		}
	}
	n := float64(len(vecs))
	for i := range pooled {
		pooled[i] /= n
	}

	var sumSq float64
	for _, x := range pooled {
		sumSq += x * x
	}
	if norm := math.Sqrt(sumSq); norm > 0 {
		for i := range pooled {
			pooled[i] /= norm
		}
	}
	return pooled
}

// legacyNovaFirstVector returns the first embedding vector of a Nova response, or nil.
func legacyNovaFirstVector(r *BedrockNovaEmbeddingResponse) []float64 {
	if r == nil || len(r.Embeddings) == 0 {
		return nil
	}
	return r.Embeddings[0].Embedding
}

// completeLegacyNovaEmbedding runs a legacy-contract Nova embedding end to end. Nova's
// SINGLE_EMBEDDING task embeds one modality per InvokeModel call, so the request is fanned
// out (see buildLegacyNovaEmbeddingRequests) and the vectors are pooled into one embedding.
// A text-only or single-image request makes one call and returns that vector unchanged.
func (provider *BedrockProvider) completeLegacyNovaEmbedding(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	novaRequests, err := buildLegacyNovaEmbeddingRequests(ctx, request)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("error building Nova embedding request", err)
	}

	path, _ := provider.getModelPathAndRegion(ctx, "invoke", request.Model, key)

	var vectors [][]float64
	var totalInputTokens int
	var haveTokens bool
	var latency time.Duration
	var providerResponseHeaders map[string]string
	var lastRawResponse []byte

	for _, novaReq := range novaRequests {
		novaReq := novaReq
		jsonData, bifrostError := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				return novaReq, nil
			})
		if bifrostError != nil {
			return nil, bifrostError
		}

		rawResponse, callLatency, headers, bifrostError := provider.completeRequest(ctx, jsonData, path, key, request.Model)
		if headers != nil {
			providerResponseHeaders = headers
		}
		if bifrostError != nil {
			if providerResponseHeaders != nil {
				ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerResponseHeaders)
			}
			return nil, providerUtils.EnrichError(ctx, bifrostError, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency+callLatency)
		}
		latency += callLatency
		lastRawResponse = rawResponse

		var novaResp BedrockNovaEmbeddingResponse
		if err := sonic.Unmarshal(rawResponse, &novaResp); err != nil {
			return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("error parsing Nova embedding response", err), jsonData, rawResponse, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
		}
		if vec := legacyNovaFirstVector(&novaResp); vec != nil {
			vectors = append(vectors, vec)
		}
		if inputTokens, ok := inputTokensFromHeaders(headers); ok {
			totalInputTokens += inputTokens
			haveTokens = true
		}
	}

	if providerResponseHeaders != nil {
		ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerResponseHeaders)
	}

	// Pool whatever succeeded; a single vector is returned as is; if no call yielded a vector,
	// fail.
	if len(vectors) == 0 {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("Nova returned no embeddings", nil), nil, lastRawResponse, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	pooled := legacyMeanPoolAndNormalize(vectors)

	bifrostResponse := &schemas.BifrostEmbeddingResponse{
		Object: "list",
		Model:  request.Model,
		Data: []schemas.EmbeddingData{
			{
				Index:     0,
				Object:    "embedding",
				Embedding: schemas.EmbeddingStruct{EmbeddingArray: pooled},
			},
		},
	}
	// A single call yields one vector of a known modality; the native invoke route reads it to
	// label the embedding. A pooled vector has no single modality and stays unlabelled.
	if len(novaRequests) == 1 && novaRequests[0].SingleEmbeddingParams != nil {
		switch {
		case novaRequests[0].SingleEmbeddingParams.Text != nil:
			bifrostResponse.Data[0].Modality = schemas.EmbeddingModalityText
		case novaRequests[0].SingleEmbeddingParams.Image != nil:
			bifrostResponse.Data[0].Modality = schemas.EmbeddingModalityImage
		}
	}
	if haveTokens {
		bifrostResponse.Usage = &schemas.BifrostLLMUsage{
			PromptTokens: totalInputTokens,
			TotalTokens:  totalInputTokens,
		}
	}

	// A base64 request is served by re-encoding the float vector, as for the other models.
	if shouldEncodeEmbeddingsAsBase64(request) {
		encodeEmbeddingsAsBase64(bifrostResponse)
	}

	bifrostResponse.ExtraFields.Latency = latency.Milliseconds()
	bifrostResponse.ExtraFields.ProviderResponseHeaders = providerResponseHeaders

	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		var rawResponseData interface{}
		if err := sonic.Unmarshal(lastRawResponse, &rawResponseData); err == nil {
			bifrostResponse.ExtraFields.RawResponse = rawResponseData
		}
	}

	return bifrostResponse, nil
}
