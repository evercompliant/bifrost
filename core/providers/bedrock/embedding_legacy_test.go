package bedrock

import (
	"context"
	"math"
	"strings"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests for the fork-only legacy embedding contract (embedding_legacy.go, plus the
// "inputImage" handling in ToBedrockTitanEmbeddingRequest).

const legacyNovaModel = "amazon.nova-2-multimodal-embeddings-v1:0"

// legacyTextInput builds the input a text or text-list request decodes to.
func legacyTextInput(texts ...string) []schemas.EmbeddingInputItem {
	items := make([]schemas.EmbeddingInputItem, len(texts))
	for i := range texts {
		t := texts[i]
		items[i] = schemas.EmbeddingInputItem{Content: schemas.EmbeddingContent{
			{Type: schemas.EmbeddingContentPartTypeText, Text: &t},
		}}
	}
	return items
}

// legacyJPEGDataURI builds a base64 image data URI; the payload is opaque to Bifrost.
func legacyJPEGDataURI(b64 string) string {
	return "data:image/jpeg;base64," + b64
}

func TestLegacyNovaPurpose(t *testing.T) {
	cases := []struct {
		inputType string
		want      string
	}{
		{"search_query", "GENERIC_RETRIEVAL"},
		{"query", "GENERIC_RETRIEVAL"},
		{"search_document", "GENERIC_INDEX"},
		{"document", "GENERIC_INDEX"},
		{"classification", "CLASSIFICATION"},
		{"clustering", "CLUSTERING"},
		{"image", "IMAGE_RETRIEVAL"},
		{"SEARCH_QUERY", "GENERIC_RETRIEVAL"}, // case-insensitive
		{"unknown", "GENERIC_INDEX"},          // default
		{"", "GENERIC_INDEX"},
	}
	for _, c := range cases {
		t.Run(c.inputType, func(t *testing.T) {
			assert.Equal(t, c.want, legacyNovaPurpose(c.inputType))
		})
	}
}

func TestResolveLegacyNovaPurpose(t *testing.T) {
	taskType := "CLUSTERING"

	t.Run("explicit embeddingPurpose wins over input_type and task_type", func(t *testing.T) {
		got := resolveLegacyNovaPurpose(map[string]interface{}{
			"embeddingPurpose": "DOCUMENT_RETRIEVAL",
			"input_type":       "search_query",
		}, &taskType)
		assert.Equal(t, "DOCUMENT_RETRIEVAL", got)
	})
	t.Run("input_type mapped when no explicit purpose", func(t *testing.T) {
		got := resolveLegacyNovaPurpose(map[string]interface{}{"input_type": "search_query"}, &taskType)
		assert.Equal(t, "GENERIC_RETRIEVAL", got)
	})
	t.Run("task_type used when neither extra param is present", func(t *testing.T) {
		assert.Equal(t, "CLUSTERING", resolveLegacyNovaPurpose(nil, &taskType))
	})
	t.Run("default when nothing is present", func(t *testing.T) {
		assert.Equal(t, "GENERIC_INDEX", resolveLegacyNovaPurpose(nil, nil))
	})
	t.Run("blank embeddingPurpose falls through", func(t *testing.T) {
		got := resolveLegacyNovaPurpose(map[string]interface{}{
			"embeddingPurpose": "",
			"input_type":       "clustering",
		}, nil)
		assert.Equal(t, "CLUSTERING", got)
	})
}

func TestCleanLegacyNovaBase64(t *testing.T) {
	assert.Equal(t, "abcDEF", cleanLegacyNovaBase64("  ab\ncD\r\nEF \t"))
	assert.Equal(t, "", cleanLegacyNovaBase64("  \n\r "))
}

func TestLegacyEmbeddingText(t *testing.T) {
	assert.Equal(t, "", legacyEmbeddingText(nil))
	assert.Equal(t, "hello", legacyEmbeddingText(legacyTextInput("hello")))
	assert.Equal(t, "a \nb \n", legacyEmbeddingText(legacyTextInput("a", "b")))

	// Media parts are ignored; only text counts.
	imageOnly := []schemas.EmbeddingInputItem{{Content: schemas.EmbeddingContent{
		{Type: schemas.EmbeddingContentPartTypeImage, Image: &schemas.EmbeddingMediaPart{}},
	}}}
	assert.Equal(t, "", legacyEmbeddingText(imageOnly))
}

func TestUseLegacyNovaEmbedding(t *testing.T) {
	assert.False(t, useLegacyNovaEmbedding(nil))
	assert.True(t, useLegacyNovaEmbedding(&schemas.BifrostEmbeddingRequest{Input: legacyTextInput("hi")}))
	assert.True(t, useLegacyNovaEmbedding(&schemas.BifrostEmbeddingRequest{}), "empty input with image extra params")

	data := "aGVsbG8="
	for _, partType := range []schemas.EmbeddingContentPartType{
		schemas.EmbeddingContentPartTypeImage,
		schemas.EmbeddingContentPartTypeAudio,
		schemas.EmbeddingContentPartTypeVideo,
		schemas.EmbeddingContentPartTypeFile,
	} {
		t.Run(string(partType), func(t *testing.T) {
			part := schemas.EmbeddingContentPart{Type: partType}
			media := &schemas.EmbeddingMediaPart{Data: &data}
			switch partType {
			case schemas.EmbeddingContentPartTypeImage:
				part.Image = media
			case schemas.EmbeddingContentPartTypeAudio:
				part.Audio = media
			case schemas.EmbeddingContentPartTypeVideo:
				part.Video = media
			default:
				part.File = media
			}
			req := &schemas.BifrostEmbeddingRequest{
				Input: []schemas.EmbeddingInputItem{{Content: schemas.EmbeddingContent{part}}},
			}
			assert.False(t, useLegacyNovaEmbedding(req), "typed media goes to upstream's converter")
		})
	}
}

func TestLegacyMeanPoolAndNormalize(t *testing.T) {
	t.Run("empty returns nil", func(t *testing.T) {
		assert.Nil(t, legacyMeanPoolAndNormalize(nil))
		assert.Nil(t, legacyMeanPoolAndNormalize([][]float64{}))
	})
	t.Run("single vector returned unchanged", func(t *testing.T) {
		v := []float64{1, 2, 3}
		assert.Equal(t, v, legacyMeanPoolAndNormalize([][]float64{v}))
	})
	t.Run("two vectors mean-pooled then L2 normalized", func(t *testing.T) {
		// mean of {3,0,0} and {0,4,0} = {1.5,2,0}; L2 norm = 2.5 -> {0.6,0.8,0}
		got := legacyMeanPoolAndNormalize([][]float64{{3, 0, 0}, {0, 4, 0}})
		require.Len(t, got, 3)
		assert.InDelta(t, 0.6, got[0], 1e-9)
		assert.InDelta(t, 0.8, got[1], 1e-9)
		assert.InDelta(t, 0.0, got[2], 1e-9)

		var norm float64
		for _, x := range got {
			norm += x * x
		}
		assert.InDelta(t, 1.0, math.Sqrt(norm), 1e-9)
	})
	t.Run("ragged vectors zero-padded to longest", func(t *testing.T) {
		got := legacyMeanPoolAndNormalize([][]float64{{2, 2}, {0}})
		require.Len(t, got, 2)
		// mean = {1,1}; L2 norm = sqrt(2) -> {1/sqrt2, 1/sqrt2}
		assert.InDelta(t, 1/math.Sqrt2, got[0], 1e-9)
		assert.InDelta(t, 1/math.Sqrt2, got[1], 1e-9)
	})
}

func TestLegacyNovaFirstVector(t *testing.T) {
	assert.Nil(t, legacyNovaFirstVector(nil))
	assert.Nil(t, legacyNovaFirstVector(&BedrockNovaEmbeddingResponse{}))
	resp := &BedrockNovaEmbeddingResponse{Embeddings: []BedrockNovaEmbedding{
		{EmbeddingType: "TEXT", Embedding: []float64{1, 2}},
	}}
	assert.Equal(t, []float64{1, 2}, legacyNovaFirstVector(resp))
}

func TestBuildLegacyNovaEmbeddingRequests(t *testing.T) {
	ctx := context.Background()
	dims := 1024

	t.Run("nil request errors", func(t *testing.T) {
		_, err := buildLegacyNovaEmbeddingRequests(ctx, nil)
		require.Error(t, err)
	})

	t.Run("no input and no images errors", func(t *testing.T) {
		_, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{})
		require.Error(t, err)
	})

	t.Run("text-only builds one text request", func(t *testing.T) {
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Model: legacyNovaModel,
			Input: legacyTextInput("hello world"),
			Params: &schemas.EmbeddingParameters{
				Dimensions:  &dims,
				ExtraParams: map[string]interface{}{"input_type": "search_document"},
			},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		p := reqs[0].SingleEmbeddingParams
		require.NotNil(t, p)
		assert.Equal(t, TaskTypeSingleEmbedding, reqs[0].TaskType)
		assert.Equal(t, BedrockNovaEmbeddingPurpose("GENERIC_INDEX"), p.EmbeddingPurpose)
		require.NotNil(t, p.EmbeddingDimension)
		assert.Equal(t, 1024, *p.EmbeddingDimension)
		require.NotNil(t, p.Text)
		assert.Equal(t, BedrockNovaTruncationMode("END"), p.Text.TruncationMode)
		require.NotNil(t, p.Text.Value)
		assert.Equal(t, "hello world", *p.Text.Value)
		assert.Nil(t, p.Image)
	})

	t.Run("a text list is joined into one text request", func(t *testing.T) {
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: legacyTextInput("a", "b"),
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		require.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.Equal(t, "a \nb \n", *reqs[0].SingleEmbeddingParams.Text.Value)
	})

	t.Run("image-only builds one image request", func(t *testing.T) {
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{
					"inputImage":  legacyJPEGDataURI("aGVsbG8="),
					"detailLevel": "DOCUMENT_IMAGE",
				},
			},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		p := reqs[0].SingleEmbeddingParams
		assert.Nil(t, p.Text)
		require.NotNil(t, p.Image)
		assert.Equal(t, "jpeg", p.Image.Format)
		require.NotNil(t, p.Image.DetailLevel)
		assert.Equal(t, BedrockNovaDetailLevel("DOCUMENT_IMAGE"), *p.Image.DetailLevel)
		require.NotNil(t, p.Image.Source.Bytes)
		assert.Equal(t, "aGVsbG8=", *p.Image.Source.Bytes)
	})

	t.Run("image-only oversized is rejected", func(t *testing.T) {
		big := strings.Repeat("A", legacyNovaMaxBase64ImageLen+1)
		_, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{"inputImage": legacyJPEGDataURI(big)}},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "image too large")
	})

	t.Run("hybrid text + one image builds two requests", func(t *testing.T) {
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: legacyTextInput("caption"),
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []string{legacyJPEGDataURI("aGVsbG8=")},
			}},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 2)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.NotNil(t, reqs[1].SingleEmbeddingParams.Image)
	})

	t.Run("hybrid text + N images builds 1+N requests", func(t *testing.T) {
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: legacyTextInput("caption"),
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []string{legacyJPEGDataURI("aGVsbG8="), legacyJPEGDataURI("d29ybGQ="), legacyJPEGDataURI("Zm9v")},
			}},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 4)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		for _, r := range reqs[1:] {
			assert.NotNil(t, r.SingleEmbeddingParams.Image)
		}
	})

	t.Run("hybrid skips an oversized image but keeps the rest", func(t *testing.T) {
		big := strings.Repeat("A", legacyNovaMaxBase64ImageLen+1)
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: legacyTextInput("caption"),
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []interface{}{legacyJPEGDataURI("aGVsbG8="), legacyJPEGDataURI(big)},
			}},
		})
		require.NoError(t, err)
		// text + one valid image; the oversized image is dropped.
		require.Len(t, reqs, 2)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.NotNil(t, reqs[1].SingleEmbeddingParams.Image)
	})

	t.Run("task_type is used when no extra param names a purpose", func(t *testing.T) {
		taskType := "CLASSIFICATION"
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input:  legacyTextInput("hi"),
			Params: &schemas.EmbeddingParameters{TaskType: &taskType},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		assert.Equal(t, BedrockNovaEmbeddingPurpose("CLASSIFICATION"), reqs[0].SingleEmbeddingParams.EmbeddingPurpose)
	})
}

func TestLegacyNovaEmbeddingWireBody(t *testing.T) {
	ctx := context.Background()
	dims := 256

	t.Run("text envelope", func(t *testing.T) {
		request := &schemas.BifrostEmbeddingRequest{
			Input: legacyTextInput("hello"),
			Params: &schemas.EmbeddingParameters{
				Dimensions:  &dims,
				ExtraParams: map[string]interface{}{"input_type": "search_query"},
			},
		}
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, request)
		require.NoError(t, err)
		require.Len(t, reqs, 1)

		wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) { return reqs[0], nil },
		)
		require.Nil(t, bifrostErr)
		assert.JSONEq(t, `{
			"taskType": "SINGLE_EMBEDDING",
			"singleEmbeddingParams": {
				"embeddingPurpose": "GENERIC_RETRIEVAL",
				"embeddingDimension": 256,
				"text": {"truncationMode": "END", "value": "hello"}
			}
		}`, string(wireBody))
	})

	t.Run("image envelope", func(t *testing.T) {
		request := &schemas.BifrostEmbeddingRequest{
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": legacyJPEGDataURI("aGVsbG8=")},
			},
		}
		reqs, err := buildLegacyNovaEmbeddingRequests(ctx, request)
		require.NoError(t, err)
		require.Len(t, reqs, 1)

		wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) { return reqs[0], nil },
		)
		require.Nil(t, bifrostErr)
		assert.JSONEq(t, `{
			"taskType": "SINGLE_EMBEDDING",
			"singleEmbeddingParams": {
				"embeddingPurpose": "GENERIC_INDEX",
				"image": {"format": "jpeg", "source": {"bytes": "aGVsbG8="}}
			}
		}`, string(wireBody))
	})
}

func TestToBedrockTitanEmbeddingRequestLegacyImage(t *testing.T) {
	t.Run("returns an error when neither text nor image is provided", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
		})
		require.Error(t, err)
		assert.Nil(t, req)
	})

	t.Run("image-only request lifts inputImage and omits text", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
			},
		})
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Equal(t, "", req.InputText)
		require.NotNil(t, req.InputImage)
		assert.Equal(t, "BASE64DATA", *req.InputImage)
		// inputImage must be lifted out of ExtraParams once consumed.
		_, present := req.ExtraParams["inputImage"]
		assert.False(t, present)
	})

	t.Run("a data URI loses its prefix", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": legacyJPEGDataURI("BASE64DATA")},
			},
		})
		require.NoError(t, err)
		require.NotNil(t, req.InputImage)
		assert.Equal(t, "BASE64DATA", *req.InputImage)
	})

	t.Run("a url is refused", func(t *testing.T) {
		_, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "https://example.com/cat.jpg"},
			},
		})
		require.Error(t, err)
	})

	t.Run("text plus image sets both fields", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: legacyTextInput("a cat"),
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, "a cat", req.InputText)
		require.NotNil(t, req.InputImage)
		assert.Equal(t, "BASE64DATA", *req.InputImage)
	})

	t.Run("an empty inputImage is treated as absent", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": ""},
			},
		})
		require.Error(t, err)
		assert.Nil(t, req)
	})

	t.Run("a non-string inputImage stays in ExtraParams for passthrough", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: legacyTextInput("hello"),
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": 123},
			},
		})
		require.NoError(t, err)
		assert.Nil(t, req.InputImage)
		assert.Equal(t, 123, req.ExtraParams["inputImage"])
	})

	t.Run("image-only wire body omits empty inputText", func(t *testing.T) {
		bifrostReq := &schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
			},
		}
		wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			context.Background(),
			bifrostReq,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				return ToBedrockTitanEmbeddingRequest(bifrostReq)
			},
		)
		require.Nil(t, bifrostErr)
		assert.NotContains(t, string(wireBody), `"inputText"`)
		assert.JSONEq(t, `{"inputImage": "BASE64DATA"}`, string(wireBody))
	})
}

func TestLegacyEmbeddingImages(t *testing.T) {
	assert.Nil(t, legacyEmbeddingImages(nil))
	assert.Nil(t, legacyEmbeddingImages(&schemas.EmbeddingParameters{}))

	t.Run("single image", func(t *testing.T) {
		got := legacyEmbeddingImages(&schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{"inputImage": "a"}})
		assert.Equal(t, []string{"a"}, got)
	})
	t.Run("single image first, then the list", func(t *testing.T) {
		got := legacyEmbeddingImages(&schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
			"inputImage":  "a",
			"inputImages": []string{"b", "c"},
		}})
		assert.Equal(t, []string{"a", "b", "c"}, got)
	})
	t.Run("a JSON round-tripped list of interfaces", func(t *testing.T) {
		got := legacyEmbeddingImages(&schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
			"inputImages": []interface{}{"b", 7, "c"},
		}})
		assert.Equal(t, []string{"b", "c"}, got)
	})
	t.Run("blank and non-string entries are skipped", func(t *testing.T) {
		got := legacyEmbeddingImages(&schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
			"inputImage":  "  ",
			"inputImages": []string{"", "b"},
		}})
		assert.Equal(t, []string{"b"}, got)
		assert.Nil(t, legacyEmbeddingImages(&schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{"inputImage": 123}}))
	})
}

// The HTTP entry points send a single space as the text of an image-only legacy request.
func TestLegacyNovaImageOnlyWithPlaceholderText(t *testing.T) {
	reqs, err := buildLegacyNovaEmbeddingRequests(context.Background(), &schemas.BifrostEmbeddingRequest{
		Input: legacyTextInput(" "),
		Params: &schemas.EmbeddingParameters{
			ExtraParams: map[string]interface{}{"inputImage": legacyJPEGDataURI("aGVsbG8=")},
		},
	})
	require.NoError(t, err)
	require.Len(t, reqs, 1, "the blank text is not embedded")
	assert.Nil(t, reqs[0].SingleEmbeddingParams.Text)
	assert.NotNil(t, reqs[0].SingleEmbeddingParams.Image)
}

func TestToBedrockTitanEmbeddingRequestLegacyImageWithPlaceholderText(t *testing.T) {
	req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
		Model: "amazon.titan-embed-image-v1",
		Input: legacyTextInput(" "),
		Params: &schemas.EmbeddingParameters{
			ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "", req.InputText, "whitespace-only text is dropped beside an image")
	require.NotNil(t, req.InputImage)
	assert.Equal(t, "BASE64DATA", *req.InputImage)
}

// Negative selection: the Nova family also holds the generative models, which must not be
// routed to an embedding path.
func TestDetermineEmbeddingModelTypeNovaNegativeSelection(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	t.Run("Nova embedding models are accepted", func(t *testing.T) {
		for _, model := range []string{
			"amazon.nova-2-multimodal-embeddings-v1:0",
			"us.amazon.nova-2-multimodal-embeddings-v1:0",
			"arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-2-multimodal-embeddings-v1:0",
		} {
			modelType, err := DetermineEmbeddingModelType(ctx, model)
			require.NoError(t, err, model)
			assert.Equal(t, "nova", modelType, model)
		}
	})

	t.Run("generative Nova models are rejected", func(t *testing.T) {
		for _, model := range []string{
			"amazon.nova-pro-v1:0",
			"amazon.nova-lite-v1:0",
			"amazon.nova-micro-v1:0",
			"amazon.nova-premier-v1:0",
			"us.amazon.nova-2-lite-v1:0",
			"amazon.nova-canvas-v1:0",
			"amazon.nova-reel-v1:0",
			"amazon.nova-sonic-v1:0",
		} {
			modelType, err := DetermineEmbeddingModelType(ctx, model)
			require.Error(t, err, model)
			assert.Empty(t, modelType, model)
			assert.Contains(t, err.Error(), "unsupported embedding model", model)
		}
	})

	t.Run("an alias is judged by the model it resolves to", func(t *testing.T) {
		embedFamily := schemas.ModelFamilyNova
		embedAlias := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		embedAlias.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
			Key: "my-vectors",
			Config: &schemas.AliasConfig{
				ModelID:     "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-2-multimodal-embeddings-v1:0",
				ModelFamily: &embedFamily,
			},
		})
		modelType, err := DetermineEmbeddingModelType(embedAlias, "my-vectors")
		require.NoError(t, err)
		assert.Equal(t, "nova", modelType)

		genFamily := schemas.ModelFamilyNova
		genAlias := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		genAlias.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
			Key: "my-chat",
			Config: &schemas.AliasConfig{
				ModelID:     "us.amazon.nova-pro-v1:0",
				ModelFamily: &genFamily,
			},
		})
		modelType, err = DetermineEmbeddingModelType(genAlias, "my-chat")
		require.Error(t, err)
		assert.Empty(t, modelType)
	})

	t.Run("other families are unaffected", func(t *testing.T) {
		for model, want := range map[string]string{
			"amazon.titan-embed-text-v2:0": "titan",
			"amazon.titan-embed-image-v1":  "titan",
			"cohere.embed-v4:0":            "cohere",
		} {
			modelType, err := DetermineEmbeddingModelType(ctx, model)
			require.NoError(t, err, model)
			assert.Equal(t, want, modelType, model)
		}
		_, err := DetermineEmbeddingModelType(ctx, "meta.llama3-8b-instruct-v1:0")
		require.Error(t, err)
	})
}
