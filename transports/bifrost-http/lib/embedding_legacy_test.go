package lib

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func legacyTextItem(text string) schemas.EmbeddingInputItem {
	t := text
	return schemas.EmbeddingInputItem{Content: schemas.EmbeddingContent{
		{Type: schemas.EmbeddingContentPartTypeText, Text: &t},
	}}
}

func TestNormalizeLegacyEmbeddingInput(t *testing.T) {
	images := map[string]interface{}{"inputImage": "aGVsbG8="}

	t.Run("no images: input is returned untouched, blank text included", func(t *testing.T) {
		in := []schemas.EmbeddingInputItem{legacyTextItem("")}
		assert.Equal(t, in, NormalizeLegacyEmbeddingInput(in, nil))
		assert.Equal(t, in, NormalizeLegacyEmbeddingInput(in, map[string]interface{}{"inputImage": 123}))
	})

	t.Run("images and an empty input get the single-space placeholder", func(t *testing.T) {
		got := NormalizeLegacyEmbeddingInput(nil, images)
		require.Len(t, got, 1)
		require.NotNil(t, got[0].Content[0].Text)
		assert.Equal(t, " ", *got[0].Content[0].Text)
	})

	t.Run("images and blank text get the placeholder", func(t *testing.T) {
		got := NormalizeLegacyEmbeddingInput([]schemas.EmbeddingInputItem{legacyTextItem(""), legacyTextItem(" \n")}, images)
		require.Len(t, got, 1)
		assert.Equal(t, " ", *got[0].Content[0].Text)
	})

	t.Run("real text is kept and blank items are dropped", func(t *testing.T) {
		got := NormalizeLegacyEmbeddingInput([]schemas.EmbeddingInputItem{legacyTextItem(""), legacyTextItem("caption")}, images)
		require.Len(t, got, 1)
		assert.Equal(t, "caption", *got[0].Content[0].Text)
	})

	t.Run("an image list is recognised in both Go shapes", func(t *testing.T) {
		for _, list := range []interface{}{[]string{"a"}, []interface{}{"a"}} {
			got := NormalizeLegacyEmbeddingInput(nil, map[string]interface{}{"inputImages": list})
			assert.Len(t, got, 1)
		}
		assert.Nil(t, NormalizeLegacyEmbeddingInput(nil, map[string]interface{}{"inputImages": []string{" "}}))
	})
}
