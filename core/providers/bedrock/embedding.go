package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// bedrockInputTokenCountHeader is the HTTP response header Bedrock uses to report input
// token counts for models — notably Cohere embed and rerank — that omit token usage from
// the response body.
const bedrockInputTokenCountHeader = "X-Amzn-Bedrock-Input-Token-Count"

// inputTokensFromHeaders extracts the X-Amzn-Bedrock-Input-Token-Count value from a provider
// response-headers map (case-insensitive, since header casing depends on the transport).
// It returns (count, true) only when the header is present and parses as a non-negative int.
func inputTokensFromHeaders(headers map[string]string) (int, bool) {
	for k, v := range headers {
		if strings.EqualFold(k, bedrockInputTokenCountHeader) {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			if err != nil || n < 0 {
				return 0, false
			}
			return n, true
		}
	}
	return 0, false
}

// ToBedrockTitanEmbeddingRequest converts a Bifrost embedding request to Bedrock Titan format
func ToBedrockTitanEmbeddingRequest(bifrostReq *schemas.BifrostEmbeddingRequest) (*BedrockTitanEmbeddingRequest, error) {
	if bifrostReq == nil {
		return nil, fmt.Errorf("bifrost embedding request is nil")
	}

	// Validate that text or image input is provided for Titan models.
	// Titan multimodal (amazon.titan-embed-image-v1) accepts an image-only
	// request, so we no longer require text. hasImage requires a non-empty
	// string value, matching the lifting contract below: treating any other
	// value (nil, non-string, empty string) as "present" would let validation
	// pass and then ship an empty {} wire body to AWS.
	hasText := bifrostReq.Input != nil && (bifrostReq.Input.Text != nil || len(bifrostReq.Input.Texts) > 0)
	var hasImage bool
	if bifrostReq.Params != nil && bifrostReq.Params.ExtraParams != nil {
		if img, ok := bifrostReq.Params.ExtraParams["inputImage"]; ok {
			if s, ok := img.(string); ok && s != "" {
				hasImage = true
			}
		}
	}
	if !hasText && !hasImage {
		return nil, fmt.Errorf("no input text or image provided for embedding")
	}

	titanReq := &BedrockTitanEmbeddingRequest{}

	// Set input text only when text is actually present; image-only requests omit this field.
	if bifrostReq.Input != nil && bifrostReq.Input.Text != nil {
		titanReq.InputText = *bifrostReq.Input.Text
	} else if bifrostReq.Input != nil && len(bifrostReq.Input.Texts) > 0 {
		var embeddingText string
		for _, text := range bifrostReq.Input.Texts {
			embeddingText += text + " \n"
		}
		titanReq.InputText = embeddingText
	}

	if bifrostReq.Params != nil {
		titanReq.Dimensions = bifrostReq.Params.Dimensions
		if normalize, ok := bifrostReq.Params.ExtraParams["normalize"]; ok {
			if b, ok := normalize.(bool); ok {
				titanReq.Normalize = &b
			}
		}

		// Lift inputImage into the typed field for guaranteed wire-format
		// inclusion (does not depend on the passthrough-extra-params context key).
		var inputImageLifted bool
		if img, ok := bifrostReq.Params.ExtraParams["inputImage"]; ok {
			if s, ok := img.(string); ok && s != "" {
				titanReq.InputImage = s
				inputImageLifted = true
			}
		}

		// Forward remaining extra params, excluding fields now represented as
		// first-class struct fields. Only exclude inputImage when it was actually
		// lifted (string case); non-string values stay in ExtraParams for passthrough.
		if len(bifrostReq.Params.ExtraParams) > 0 {
			extra := make(map[string]interface{})
			for k, v := range bifrostReq.Params.ExtraParams {
				if k == "normalize" {
					continue
				}
				if k == "inputImage" && inputImageLifted {
					continue
				}
				extra[k] = v
			}
			if len(extra) > 0 {
				titanReq.ExtraParams = extra
			}
		}
	}

	return titanReq, nil
}

// ToBifrostEmbeddingResponse converts a Bedrock Titan embedding response to Bifrost format
func (response *BedrockTitanEmbeddingResponse) ToBifrostEmbeddingResponse() *schemas.BifrostEmbeddingResponse {
	if response == nil {
		return nil
	}

	bifrostResponse := &schemas.BifrostEmbeddingResponse{
		Object: "list",
		Data: []schemas.EmbeddingData{
			{
				Index:  0,
				Object: "embedding",
				Embedding: schemas.EmbeddingStruct{
					EmbeddingArray: response.Embedding,
				},
			},
		},
		Usage: &schemas.BifrostLLMUsage{
			PromptTokens: response.InputTextTokenCount,
			TotalTokens:  response.InputTextTokenCount,
		},
	}

	return bifrostResponse
}

// ToBedrockCohereEmbeddingRequest converts a Bifrost embedding request to Bedrock Cohere format.
// Unlike the direct Cohere API, Bedrock does not accept a "model" field in the request body.
func ToBedrockCohereEmbeddingRequest(bifrostReq *schemas.BifrostEmbeddingRequest) (*BedrockCohereEmbeddingRequest, error) {
	if bifrostReq == nil {
		return nil, fmt.Errorf("bifrost embedding request is nil")
	}
	if bifrostReq.Input == nil || (bifrostReq.Input.Text == nil && len(bifrostReq.Input.Texts) == 0) {
		return nil, fmt.Errorf("no input provided for embedding")
	}

	req := &BedrockCohereEmbeddingRequest{}

	// Map texts
	if bifrostReq.Input.Text != nil {
		req.Texts = []string{*bifrostReq.Input.Text}
	} else if len(bifrostReq.Input.Texts) > 0 {
		req.Texts = bifrostReq.Input.Texts
	}

	if bifrostReq.Params != nil {
		extra := make(map[string]interface{}, len(bifrostReq.Params.ExtraParams))
		for k, v := range bifrostReq.Params.ExtraParams {
			extra[k] = v
		}

		if v, ok := extra["input_type"]; ok {
			if s, ok := v.(string); ok {
				req.InputType = s
				delete(extra, "input_type")
			}
		}
		if v, ok := extra["truncate"]; ok {
			if s, ok := v.(string); ok {
				req.Truncate = &s
				delete(extra, "truncate")
			}
		}
		if v, ok := extra["embedding_types"]; ok {
			if ss, ok := v.([]string); ok {
				req.EmbeddingTypes = ss
				delete(extra, "embedding_types")
			}
		}
		if v, ok := extra["images"]; ok {
			if ss, ok := v.([]string); ok {
				req.Images = ss
				delete(extra, "images")
			}
		}
		if v, ok := extra["inputs"]; ok {
			if inputs, ok := v.([]BedrockCohereEmbeddingInput); ok {
				req.Inputs = inputs
				delete(extra, "inputs")
			}
		}
		if v, ok := extra["max_tokens"]; ok {
			switch n := v.(type) {
			case int:
				req.MaxTokens = &n
				delete(extra, "max_tokens")
			case float64:
				i := int(n)
				req.MaxTokens = &i
				delete(extra, "max_tokens")
			}
		}
		if bifrostReq.Params.Dimensions != nil {
			req.OutputDimension = bifrostReq.Params.Dimensions
		}
		if len(extra) > 0 {
			req.ExtraParams = extra
		}
	}

	return req, nil
}

// DetermineEmbeddingModelType determines the embedding model type for the
// current attempt. It consults the resolved alias family first
// (model_family / model_name / model_id / alias key) and falls back to the
// substring detectors against the wire model — so an alias to an opaque
// Bedrock deployment that's tagged with the right family routes correctly.
func DetermineEmbeddingModelType(ctx *schemas.BifrostContext, model string) (string, error) {
	switch {
	// Explicit match for the multimodal embeddings model — it contains "nova"
	// but not the "lite"/"sonic" markers some Nova-2 detectors gate on, so be
	// unambiguous here rather than relying on family substring resolution.
	case strings.Contains(model, "nova") && strings.Contains(model, "embed"):
		return "nova", nil
	case schemas.IsNovaModelFamily(ctx, model):
		return "nova", nil
	case schemas.IsTitanModelFamily(ctx, model):
		return "titan", nil
	case schemas.IsCohereModelFamily(ctx, model):
		return "cohere", nil
	default:
		return "", fmt.Errorf("unsupported embedding model: %s", model)
	}
}

// ToBifrostEmbeddingResponse converts a BedrockCohereEmbeddingResponse to Bifrost format.
// Bedrock returns embeddings as a raw [][]float32 when response_type is "embeddings_floats"
// (the default, when no embedding_types are requested), and as a typed object when
// response_type is "embeddings_by_type".
func (r *BedrockCohereEmbeddingResponse) ToBifrostEmbeddingResponse() (*schemas.BifrostEmbeddingResponse, error) {
	if r == nil {
		return nil, fmt.Errorf("nil Bedrock Cohere embedding response")
	}

	bifrostResponse := &schemas.BifrostEmbeddingResponse{Object: "list"}

	switch r.ResponseType {
	case "embeddings_by_type":
		// Object form: {"float": [[...]], "int8": [[...]], "uint8": [[...]], "binary": [[...]], "ubinary": [[...]], "base64": [...]}
		var typed struct {
			Float   [][]float32 `json:"float"`
			Base64  []string    `json:"base64"`
			Int8    [][]int8    `json:"int8"`
			Uint8   [][]int32   `json:"uint8"` // int32 avoids []byte→base64 JSON issue
			Binary  [][]int8    `json:"binary"`
			Ubinary [][]int32   `json:"ubinary"` // int32 avoids []byte→base64 JSON issue
		}
		if err := json.Unmarshal(r.Embeddings, &typed); err != nil {
			return nil, fmt.Errorf("error parsing embeddings_by_type: %w", err)
		}
		if typed.Float != nil {
			for i, emb := range typed.Float {
				float64Emb := make([]float64, len(emb))
				for j, v := range emb {
					float64Emb[j] = float64(v)
				}
				bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
					Object:    "embedding",
					Index:     i,
					Embedding: schemas.EmbeddingStruct{EmbeddingArray: float64Emb},
				})
			}
		}
		if typed.Base64 != nil {
			for i, emb := range typed.Base64 {
				e := emb
				bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
					Object:    "embedding",
					Index:     i,
					Embedding: schemas.EmbeddingStruct{EmbeddingStr: &e},
				})
			}
		}
		for i, emb := range typed.Int8 {
			bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
				Object:    "embedding",
				Index:     i,
				Embedding: schemas.EmbeddingStruct{EmbeddingInt8Array: emb},
			})
		}
		for i, emb := range typed.Binary {
			bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
				Object:    "embedding",
				Index:     i,
				Embedding: schemas.EmbeddingStruct{EmbeddingInt8Array: emb},
			})
		}
		for i, emb := range typed.Uint8 {
			bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
				Object:    "embedding",
				Index:     i,
				Embedding: schemas.EmbeddingStruct{EmbeddingInt32Array: emb},
			})
		}
		for i, emb := range typed.Ubinary {
			bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
				Object:    "embedding",
				Index:     i,
				Embedding: schemas.EmbeddingStruct{EmbeddingInt32Array: emb},
			})
		}

	default:
		// Default / "embeddings_floats": raw array form [[...], [...]]
		var floats [][]float32
		if err := json.Unmarshal(r.Embeddings, &floats); err != nil {
			return nil, fmt.Errorf("error parsing embeddings_floats: %w", err)
		}
		for i, emb := range floats {
			float64Emb := make([]float64, len(emb))
			for j, v := range emb {
				float64Emb[j] = float64(v)
			}
			bifrostResponse.Data = append(bifrostResponse.Data, schemas.EmbeddingData{
				Object:    "embedding",
				Index:     i,
				Embedding: schemas.EmbeddingStruct{EmbeddingArray: float64Emb},
			})
		}
	}

	return bifrostResponse, nil
}

// ==================== NOVA MULTIMODAL EMBEDDINGS ====================

// ExtraParams keys understood by the Nova embedding request builder.
//
//	embeddingPurpose : raw Nova purpose enum (GENERIC_INDEX, GENERIC_RETRIEVAL, ...),
//	                   takes precedence over input_type-derived mapping.
//	input_type       : Cohere-style hint (search_document / search_query / ...)
//	                   mapped to a Nova purpose when embeddingPurpose is absent.
//	truncationMode   : text truncation mode (START | END | NONE); defaults to END.
//	detailLevel      : image detail level (STANDARD_IMAGE | DOCUMENT_IMAGE).
//	inputImage       : single image (data-URI, http(s) URL, or raw base64).
//	inputImages      : []string of images for the multi-image (hybrid) case.
const (
	novaExtraParamEmbeddingPurpose = "embeddingPurpose"
	novaExtraParamInputType        = "input_type"
	novaExtraParamTruncationMode   = "truncationMode"
	novaExtraParamDetailLevel      = "detailLevel"
	novaExtraParamInputImage       = "inputImage"
	novaExtraParamInputImages      = "inputImages"
)

// novaMaxBase64ImageLen mirrors the Python reference: Nova accepts ~5MB of raw
// image, which is ~6.7MB base64-encoded. Reject anything larger than ~6.5MB of
// encoded bytes as "image too large" rather than forwarding a doomed request.
const novaMaxBase64ImageLen = 6_500_000

// novaDefaultPurpose is used when neither embeddingPurpose nor input_type is set.
const novaDefaultPurpose = "GENERIC_INDEX"

// novaPurpose maps a Cohere-style input_type hint to a Nova embeddingPurpose.
// Index-time inputs map to GENERIC_INDEX and query-time inputs to
// GENERIC_RETRIEVAL, mirroring the Python _nova_purpose(input_type) helper.
// Unknown/empty values fall back to novaDefaultPurpose.
func novaPurpose(inputType string) string {
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
		return novaDefaultPurpose
	}
}

// resolveNovaPurpose picks the embeddingPurpose for a request: an explicit
// embeddingPurpose ExtraParam wins (raw enum passthrough), otherwise the
// input_type hint is mapped, otherwise the default.
func resolveNovaPurpose(extra map[string]interface{}) string {
	if v, ok := extra[novaExtraParamEmbeddingPurpose]; ok {
		if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
			return s
		}
	}
	if v, ok := extra[novaExtraParamInputType]; ok {
		if s, ok := v.(string); ok {
			return novaPurpose(s)
		}
	}
	return novaDefaultPurpose
}

// cleanNovaBase64 strips whitespace (newlines, carriage returns, spaces) from a
// base64 payload, mirroring the Python image cleaning step.
func cleanNovaBase64(b64 string) string {
	replacer := strings.NewReplacer("\n", "", "\r", "", " ", "", "\t", "")
	return strings.TrimSpace(replacer.Replace(b64))
}

// novaCollectImageInputs gathers image inputs from ExtraParams, accepting either
// a single inputImage string or an inputImages []string (or []interface{} of
// strings, since ExtraParams often round-trips through JSON).
func novaCollectImageInputs(extra map[string]interface{}) []string {
	var images []string
	if v, ok := extra[novaExtraParamInputImage]; ok {
		if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
			images = append(images, s)
		}
	}
	switch v := extra[novaExtraParamInputImages].(type) {
	case []string:
		images = append(images, v...)
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				images = append(images, s)
			}
		}
	}
	return images
}

// resolveNovaImage normalizes a single image input (data-URI, URL, or raw
// base64) into Nova's format + base64-bytes shape, then applies base64 cleaning
// and the encoded-size cap. It returns an error for invalid/oversized images so
// callers can decide whether to skip (hybrid) or fail (image-only).
func resolveNovaImage(ctx context.Context, image, detailLevel string) (*BedrockNovaEmbeddingImage, error) {
	src, err := convertImageToBedrockSource(ctx, image)
	if err != nil {
		return nil, fmt.Errorf("invalid image input: %w", err)
	}
	if src.Source.Bytes == nil {
		return nil, fmt.Errorf("invalid image input: no image bytes")
	}
	clean := cleanNovaBase64(*src.Source.Bytes)
	if clean == "" {
		return nil, fmt.Errorf("invalid image input: empty image bytes")
	}
	if len(clean) > novaMaxBase64ImageLen {
		return nil, fmt.Errorf("image too large: %d base64 bytes exceeds limit of %d", len(clean), novaMaxBase64ImageLen)
	}
	return &BedrockNovaEmbeddingImage{
		DetailLevel: detailLevel,
		Format:      src.Format,
		Source:      BedrockImageSourceData{Bytes: &clean},
	}, nil
}

// buildNovaEmbeddingRequests builds the ordered list of Nova single-embedding
// requests for a Bifrost embedding request: one text request (when text is
// present) followed by one request per valid image. This mirrors the Python
// reference: text-only → 1 call, image-only → 1 call, hybrid → 1 text call plus
// one call per valid image (invalid/oversized images are skipped). ctx is
// required to normalize image inputs (which may be URLs that need fetching).
//
// It returns an error when nothing embeddable remains — e.g. no text and every
// image was invalid/oversized — surfacing the last image error (such as "image
// too large") so the caller gets an actionable message.
func buildNovaEmbeddingRequests(ctx context.Context, bifrostReq *schemas.BifrostEmbeddingRequest) ([]*BedrockNovaEmbeddingRequest, error) {
	if bifrostReq == nil {
		return nil, fmt.Errorf("bifrost embedding request is nil")
	}
	if bifrostReq.Input == nil {
		return nil, fmt.Errorf("no input provided for embedding")
	}

	var extra map[string]interface{}
	var dimensions *int
	if bifrostReq.Params != nil {
		extra = bifrostReq.Params.ExtraParams
		dimensions = bifrostReq.Params.Dimensions
	}
	purpose := resolveNovaPurpose(extra)

	truncationMode := "END"
	if v, ok := extra[novaExtraParamTruncationMode]; ok {
		if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
			truncationMode = s
		}
	}
	var detailLevel string
	if v, ok := extra[novaExtraParamDetailLevel]; ok {
		if s, ok := v.(string); ok {
			detailLevel = s
		}
	}

	// Assemble text from Text or (space-joined) Texts, matching Titan behavior.
	var text string
	if bifrostReq.Input.Text != nil {
		text = *bifrostReq.Input.Text
	} else if len(bifrostReq.Input.Texts) > 0 {
		var b strings.Builder
		for _, t := range bifrostReq.Input.Texts {
			b.WriteString(t)
			b.WriteString(" \n")
		}
		text = b.String()
	}

	var requests []*BedrockNovaEmbeddingRequest
	if strings.TrimSpace(text) != "" {
		requests = append(requests, &BedrockNovaEmbeddingRequest{
			TaskType: NovaTaskTypeSingleEmbedding,
			SingleEmbeddingParams: BedrockNovaSingleEmbeddingParams{
				EmbeddingPurpose:   purpose,
				EmbeddingDimension: dimensions,
				Text:               &BedrockNovaEmbeddingText{TruncationMode: truncationMode, Value: text},
			},
		})
	}

	var lastImageErr error
	for _, image := range novaCollectImageInputs(extra) {
		img, err := resolveNovaImage(ctx, image, detailLevel)
		if err != nil {
			// Skip invalid/oversized images (mirrors _is_valid_image_b64 filtering
			// and the ImageTooLarge rejection); remember the reason in case nothing
			// embeddable is left.
			lastImageErr = err
			continue
		}
		requests = append(requests, &BedrockNovaEmbeddingRequest{
			TaskType: NovaTaskTypeSingleEmbedding,
			SingleEmbeddingParams: BedrockNovaSingleEmbeddingParams{
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

// meanPoolAndNormalize element-wise mean-pools the given vectors and L2
// normalizes the result, mirroring the Python _nova_embed_product pooling.
// It returns nil for no vectors, the single vector unchanged for exactly one,
// and the pooled+normalized vector otherwise. Vectors shorter than the first
// are treated as zero-padded so a truncated/odd response cannot panic.
func meanPoolAndNormalize(vecs [][]float64) []float64 {
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
	norm := math.Sqrt(sumSq)
	if norm > 0 {
		for i := range pooled {
			pooled[i] /= norm
		}
	}
	return pooled
}
