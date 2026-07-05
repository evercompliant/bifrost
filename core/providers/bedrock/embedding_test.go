package bedrock

import (
	"context"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToBedrockCohereEmbeddingRequest(t *testing.T) {
	t.Run("returns error for nil request", func(t *testing.T) {
		req, err := ToBedrockCohereEmbeddingRequest(nil)
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "nil")
	})

	t.Run("returns error for missing input", func(t *testing.T) {
		req, err := ToBedrockCohereEmbeddingRequest(&schemas.BifrostEmbeddingRequest{})
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "no input")
	})

	t.Run("returns error for non-nil but empty input", func(t *testing.T) {
		req, err := ToBedrockCohereEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{},
		})
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "no input")
	})

	t.Run("single text strips model and extracts typed params", func(t *testing.T) {
		text := "hello"
		truncate := "RIGHT"
		dimensions := 512
		bifrostReq := &schemas.BifrostEmbeddingRequest{
			Model: "cohere.embed-english-v3",
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{
				Dimensions: &dimensions,
				ExtraParams: map[string]interface{}{
					"input_type":      "search_query",
					"embedding_types": []string{"float"},
					"truncate":        truncate,
					"max_tokens":      float64(128),
					"trace_id":        "req-123",
				},
			},
		}

		req, err := ToBedrockCohereEmbeddingRequest(bifrostReq)
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Equal(t, "search_query", req.InputType)
		assert.Equal(t, []string{"hello"}, req.Texts)
		assert.Equal(t, []string{"float"}, req.EmbeddingTypes)
		assert.Equal(t, &dimensions, req.OutputDimension)
		assert.Equal(t, 128, *req.MaxTokens)
		require.NotNil(t, req.Truncate)
		assert.Equal(t, truncate, *req.Truncate)
		assert.Equal(t, map[string]interface{}{"trace_id": "req-123"}, req.ExtraParams)
	})

	t.Run("multiple texts preserve bedrock body shape", func(t *testing.T) {
		bifrostReq := &schemas.BifrostEmbeddingRequest{
			Model: "cohere.embed-multilingual-v3",
			Input: &schemas.EmbeddingInput{Texts: []string{"hello", "world"}},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{
					"input_type": "search_document",
				},
			},
		}

		req, err := ToBedrockCohereEmbeddingRequest(bifrostReq)
		require.NoError(t, err)
		assert.Equal(t, []string{"hello", "world"}, req.Texts)
		assert.Equal(t, "search_document", req.InputType)
	})
}

func TestToBedrockCohereEmbeddingRequestBodyOmitsModel(t *testing.T) {
	text := "hello"
	bifrostReq := &schemas.BifrostEmbeddingRequest{
		Model: "cohere.embed-english-v3",
		Input: &schemas.EmbeddingInput{Text: &text},
		Params: &schemas.EmbeddingParameters{
			ExtraParams: map[string]interface{}{
				"input_type":      "search_document",
				"embedding_types": []string{"float"},
			},
		},
	}

	wireBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		context.Background(),
		bifrostReq,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToBedrockCohereEmbeddingRequest(bifrostReq)
		},
	)
	require.Nil(t, bifrostErr)
	assert.NotContains(t, string(wireBody), `"model"`)
	assert.JSONEq(t, `{
		"input_type": "search_document",
		"texts": ["hello"],
		"embedding_types": ["float"]
	}`, string(wireBody))
}

func TestToBedrockTitanEmbeddingRequest(t *testing.T) {
	t.Run("returns error for nil request", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(nil)
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "nil")
	})

	t.Run("returns error when neither text nor image provided", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Input: &schemas.EmbeddingInput{},
		})
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "no input text or image")
	})

	t.Run("text-only request sets inputText and no image", func(t *testing.T) {
		text := "hello"
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-text-v2:0",
			Input: &schemas.EmbeddingInput{Text: &text},
		})
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Equal(t, "hello", req.InputText)
		assert.Equal(t, "", req.InputImage)
	})

	t.Run("image-only request lifts inputImage and omits text", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: &schemas.EmbeddingInput{},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
			},
		})
		require.NoError(t, err)
		require.NotNil(t, req)
		assert.Equal(t, "", req.InputText)
		assert.Equal(t, "BASE64DATA", req.InputImage)
		// inputImage must be lifted out of ExtraParams once consumed.
		_, present := req.ExtraParams["inputImage"]
		assert.False(t, present)
	})

	t.Run("text plus image sets both fields", func(t *testing.T) {
		text := "a cat"
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": "BASE64DATA"},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, "a cat", req.InputText)
		assert.Equal(t, "BASE64DATA", req.InputImage)
	})

	t.Run("empty-string inputImage is treated as absent", func(t *testing.T) {
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: &schemas.EmbeddingInput{},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": ""},
			},
		})
		require.Error(t, err)
		assert.Nil(t, req)
		assert.Contains(t, err.Error(), "no input text or image")
	})

	t.Run("non-string inputImage stays in ExtraParams for passthrough", func(t *testing.T) {
		text := "hello"
		req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: &schemas.EmbeddingInput{Text: &text},
			Params: &schemas.EmbeddingParameters{
				ExtraParams: map[string]interface{}{"inputImage": 123},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, "", req.InputImage)
		assert.Equal(t, 123, req.ExtraParams["inputImage"])
	})

	t.Run("image-only wire body omits empty inputText", func(t *testing.T) {
		bifrostReq := &schemas.BifrostEmbeddingRequest{
			Model: "amazon.titan-embed-image-v1",
			Input: &schemas.EmbeddingInput{},
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
