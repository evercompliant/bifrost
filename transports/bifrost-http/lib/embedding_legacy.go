package lib

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// Fork-only (everc) shim for the legacy image-via-extra-params embedding contract.
//
// Before upstream 2.3.0 an embeddings request could carry its images as the "inputImage" /
// "inputImages" extra params next to a text input that was empty, blank or a placeholder, and
// an image-only request was valid. 2.3.0 rejects an empty input and an empty text part at the
// HTTP entry points and again in core. The Bedrock provider (embedding_legacy.go in the forked
// core/providers/bedrock) still reads the images, but it only sees requests that get past those
// checks, so this shim keeps them from tripping: it drops blank text items and, when nothing
// else is left, substitutes a single-space text item. The provider treats whitespace-only text
// as absent, so the request stays image-only.

// legacyEmbeddingImageParams are the extra params the shim recognises.
var legacyEmbeddingImageParams = [...]string{"inputImage", "inputImages"}

// hasLegacyEmbeddingImages reports whether the extra params carry at least one non-blank image.
func hasLegacyEmbeddingImages(extra map[string]interface{}) bool {
	for _, name := range legacyEmbeddingImageParams {
		switch v := extra[name].(type) {
		case string:
			if strings.TrimSpace(v) != "" {
				return true
			}
		case []string:
			for _, s := range v {
				if strings.TrimSpace(s) != "" {
					return true
				}
			}
		case []interface{}:
			for _, item := range v {
				if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
					return true
				}
			}
		}
	}
	return false
}

// isBlankTextEmbeddingItem reports whether an item holds only text parts with blank text.
func isBlankTextEmbeddingItem(item schemas.EmbeddingInputItem) bool {
	if len(item.Content) == 0 {
		return true
	}
	for _, part := range item.Content {
		if part.Type != schemas.EmbeddingContentPartTypeText {
			return false
		}
		if part.Text != nil && strings.TrimSpace(*part.Text) != "" {
			return false
		}
	}
	return true
}

// NormalizeLegacyEmbeddingInput returns the input unchanged unless the extra params carry
// legacy images. In that case it drops the blank text items and, if none remain, returns one
// single-space text item, so the request passes the "input is required" and "text must not be
// empty" checks.
func NormalizeLegacyEmbeddingInput(input []schemas.EmbeddingInputItem, extra map[string]interface{}) []schemas.EmbeddingInputItem {
	if !hasLegacyEmbeddingImages(extra) {
		return input
	}
	kept := make([]schemas.EmbeddingInputItem, 0, len(input))
	for _, item := range input {
		if !isBlankTextEmbeddingItem(item) {
			kept = append(kept, item)
		}
	}
	if len(kept) == 0 {
		placeholder := " "
		kept = append(kept, schemas.EmbeddingInputItem{Content: schemas.EmbeddingContent{
			{Type: schemas.EmbeddingContentPartTypeText, Text: &placeholder},
		}})
	}
	return kept
}
