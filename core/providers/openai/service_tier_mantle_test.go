package openai

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestIsKnownGoodServiceTier(t *testing.T) {
	tests := []struct {
		name     string
		provider schemas.ModelProvider
		model    string
		tier     schemas.BifrostServiceTier
		want     bool
	}{
		{"mantle gemma priority", schemas.BedrockMantle, "google.gemma-4-26b-a4b", schemas.BifrostServiceTierPriority, true},
		{"mantle gemma flex", schemas.BedrockMantle, "google.gemma-4-26b-a4b", schemas.BifrostServiceTierFlex, true},
		{"mantle gemma default", schemas.BedrockMantle, "google.gemma-4-26b-a4b", schemas.BifrostServiceTierDefault, false},
		{"different mantle model", schemas.BedrockMantle, "some-other-model", schemas.BifrostServiceTierPriority, false},
		{"classic bedrock", schemas.Bedrock, "google.gemma-4-26b-a4b", schemas.BifrostServiceTierPriority, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isKnownGoodServiceTier(tt.provider, tt.model, tt.tier)
			if got != tt.want {
				t.Fatalf("isKnownGoodServiceTier(%q, %q, %q) = %v, want %v", tt.provider, tt.model, tt.tier, got, tt.want)
			}
		})
	}
}
