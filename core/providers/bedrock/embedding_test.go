package bedrock

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetermineEmbeddingModelType(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantType  string
		wantError bool
	}{
		{
			name:     "titan text model",
			model:    "amazon.titan-embed-text-v1",
			wantType: "titan",
		},
		{
			name:     "titan image model",
			model:    "amazon.titan-embed-image-v1",
			wantType: "titan",
		},
		{
			name:     "nova multimodal embeddings model",
			model:    "amazon.nova-2-multimodal-embeddings-v1:0",
			wantType: "titan",
		},
		{
			name:     "cohere embed model",
			model:    "cohere.embed-english-v3",
			wantType: "cohere",
		},
		{
			name:      "unsupported model",
			model:     "unknown.embedding-model-v1",
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			modelType, err := DetermineEmbeddingModelType(tc.model)
			if tc.wantError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tc.wantType, modelType)
		})
	}
}

func TestToBedrockTitanEmbeddingRequestPreservesInputImageExtraParam(t *testing.T) {
	inputText := "multimodal embedding"
	inputImage := "iVBORw0KGgoAAAANSUhEUgAA"
	normalize := true

	req, err := ToBedrockTitanEmbeddingRequest(&schemas.BifrostEmbeddingRequest{
		Input: &schemas.EmbeddingInput{
			Text: &inputText,
		},
		Params: &schemas.EmbeddingParameters{
			ExtraParams: map[string]interface{}{
				"inputImage": inputImage,
				"normalize":  normalize,
			},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, req)
	assert.Equal(t, inputText, req.InputText)
	require.NotNil(t, req.Normalize)
	assert.True(t, *req.Normalize)
	require.NotNil(t, req.ExtraParams)
	assert.Equal(t, inputImage, req.ExtraParams["inputImage"])
	assert.NotContains(t, req.ExtraParams, "normalize")
}
