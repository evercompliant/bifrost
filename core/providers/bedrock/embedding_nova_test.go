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

// jpegDataURI builds a base64 image data URI whose decoded payload is irrelevant
// (Nova image bytes are opaque to Bifrost); b64 is used verbatim as the payload.
func jpegDataURI(b64 string) string {
	return "data:image/jpeg;base64," + b64
}

func TestNovaPurpose(t *testing.T) {
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
			assert.Equal(t, c.want, novaPurpose(c.inputType))
		})
	}
}

func TestResolveNovaPurpose(t *testing.T) {
	t.Run("explicit embeddingPurpose wins over input_type", func(t *testing.T) {
		got := resolveNovaPurpose(map[string]interface{}{
			"embeddingPurpose": "DOCUMENT_RETRIEVAL",
			"input_type":       "search_query",
		})
		assert.Equal(t, "DOCUMENT_RETRIEVAL", got)
	})
	t.Run("input_type mapped when no explicit purpose", func(t *testing.T) {
		got := resolveNovaPurpose(map[string]interface{}{"input_type": "search_query"})
		assert.Equal(t, "GENERIC_RETRIEVAL", got)
	})
	t.Run("default when neither present", func(t *testing.T) {
		assert.Equal(t, "GENERIC_INDEX", resolveNovaPurpose(nil))
	})
	t.Run("blank embeddingPurpose falls through", func(t *testing.T) {
		got := resolveNovaPurpose(map[string]interface{}{
			"embeddingPurpose": "",
			"input_type":       "clustering",
		})
		assert.Equal(t, "CLUSTERING", got)
	})
}

func TestCleanNovaBase64(t *testing.T) {
	assert.Equal(t, "abcDEF", cleanNovaBase64("  ab\ncD\r\nEF \t"))
	assert.Equal(t, "", cleanNovaBase64("  \n\r "))
}

func TestMeanPoolAndNormalize(t *testing.T) {
	t.Run("empty returns nil", func(t *testing.T) {
		assert.Nil(t, meanPoolAndNormalize(nil))
		assert.Nil(t, meanPoolAndNormalize([][]float64{}))
	})
	t.Run("single vector returned unchanged", func(t *testing.T) {
		v := []float64{1, 2, 3}
		assert.Equal(t, v, meanPoolAndNormalize([][]float64{v}))
	})
	t.Run("two vectors mean-pooled then L2 normalized", func(t *testing.T) {
		// mean of {3,0,0} and {0,4,0} = {1.5,2,0}; L2 norm = 2.5 → {0.6,0.8,0}
		got := meanPoolAndNormalize([][]float64{{3, 0, 0}, {0, 4, 0}})
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
		got := meanPoolAndNormalize([][]float64{{2, 2}, {0}})
		require.Len(t, got, 2)
		// mean = {1,1}; L2 norm = sqrt(2) → {1/sqrt2, 1/sqrt2}
		assert.InDelta(t, 1/math.Sqrt2, got[0], 1e-9)
		assert.InDelta(t, 1/math.Sqrt2, got[1], 1e-9)
	})
}

func TestFirstEmbedding(t *testing.T) {
	assert.Nil(t, (*BedrockNovaEmbeddingResponse)(nil).FirstEmbedding())
	assert.Nil(t, (&BedrockNovaEmbeddingResponse{}).FirstEmbedding())
	resp := &BedrockNovaEmbeddingResponse{Embeddings: []BedrockNovaEmbeddingItem{
		{EmbeddingType: "TEXT", Embedding: []float64{1, 2}},
	}}
	assert.Equal(t, []float64{1, 2}, resp.FirstEmbedding())
}

func TestBuildNovaEmbeddingRequests(t *testing.T) {
	ctx := context.Background()
	dims := 1024

	t.Run("nil request errors", func(t *testing.T) {
		_, err := buildNovaEmbeddingRequests(ctx, nil)
		require.Error(t, err)
	})

	t.Run("no input errors", func(t *testing.T) {
		_, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{Input: &schemas.EmbeddingInput{}})
		require.Error(t, err)
	})

	t.Run("text-only builds one text request", func(t *testing.T) {
		text := "hello world"
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Model: "amazon.nova-2-multimodal-embeddings-v1:0",
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{
				Dimensions:  &dims,
				ExtraParams: map[string]interface{}{"input_type": "search_document"},
			},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		p := reqs[0].SingleEmbeddingParams
		assert.Equal(t, NovaTaskTypeSingleEmbedding, reqs[0].TaskType)
		assert.Equal(t, "GENERIC_INDEX", p.EmbeddingPurpose)
		require.NotNil(t, p.EmbeddingDimension)
		assert.Equal(t, 1024, *p.EmbeddingDimension)
		require.NotNil(t, p.Text)
		assert.Equal(t, "END", p.Text.TruncationMode)
		assert.Equal(t, "hello world", p.Text.Value)
		assert.Nil(t, p.Image)
	})

	t.Run("texts slice joined into one text request", func(t *testing.T) {
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Texts: []string{"a", "b"}},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)
		require.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.Equal(t, "a \nb \n", reqs[0].SingleEmbeddingParams.Text.Value)
	})

	t.Run("image-only builds one image request", func(t *testing.T) {
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Texts: []string{}},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{
					"inputImage":  jpegDataURI("aGVsbG8="),
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
		assert.Equal(t, "DOCUMENT_IMAGE", p.Image.DetailLevel)
		require.NotNil(t, p.Image.Source.Bytes)
		assert.Equal(t, "aGVsbG8=", *p.Image.Source.Bytes)
	})

	t.Run("image-only oversized is rejected", func(t *testing.T) {
		big := strings.Repeat("A", novaMaxBase64ImageLen+1)
		_, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input:  &schemas.EmbeddingInput{Texts: []string{}},
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{"inputImage": jpegDataURI(big)}},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "image too large")
	})

	t.Run("hybrid text + one image builds two requests", func(t *testing.T) {
		text := "caption"
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []string{jpegDataURI("aGVsbG8=")},
			}},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 2)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.NotNil(t, reqs[1].SingleEmbeddingParams.Image)
	})

	t.Run("hybrid text + N images builds 1+N requests", func(t *testing.T) {
		text := "caption"
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []string{jpegDataURI("aGVsbG8="), jpegDataURI("d29ybGQ="), jpegDataURI("Zm9v")},
			}},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 4)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		for _, r := range reqs[1:] {
			assert.NotNil(t, r.SingleEmbeddingParams.Image)
		}
	})

	t.Run("hybrid skips oversized image but keeps the rest", func(t *testing.T) {
		text := "caption"
		big := strings.Repeat("A", novaMaxBase64ImageLen+1)
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{ExtraParams: map[string]interface{}{
				"inputImages": []interface{}{jpegDataURI("aGVsbG8="), jpegDataURI(big)},
			}},
		})
		require.NoError(t, err)
		// text + one valid image; oversized image dropped.
		require.Len(t, reqs, 2)
		assert.NotNil(t, reqs[0].SingleEmbeddingParams.Text)
		assert.NotNil(t, reqs[1].SingleEmbeddingParams.Image)
	})
}

func TestNovaEmbeddingWireBody(t *testing.T) {
	ctx := context.Background()
	dims := 256

	t.Run("text envelope", func(t *testing.T) {
		text := "hello"
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{
				Dimensions:  &dims,
				ExtraParams: map[string]interface{}{"input_type": "search_query"},
			},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)

		wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			&schemas.BifrostEmbeddingRequest{Input: &schemas.EmbeddingInput{Text: &text}},
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
		reqs, err := buildNovaEmbeddingRequests(ctx, &schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{Texts: []string{}},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": jpegDataURI("aGVsbG8=")},
			},
		})
		require.NoError(t, err)
		require.Len(t, reqs, 1)

		wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			&schemas.BifrostEmbeddingRequest{Input: &schemas.EmbeddingInput{Texts: []string{"x"}}},
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

func TestDetermineEmbeddingModelTypeNova(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	mt, err := DetermineEmbeddingModelType(ctx, "amazon.nova-2-multimodal-embeddings-v1:0")
	require.NoError(t, err)
	assert.Equal(t, "nova", mt)
}
