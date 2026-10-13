package integrations

import (
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
)

// legacyEmbeddingRequest applies the fork-only legacy "inputImage"/"inputImages" shim to a
// converted embedding request (see lib/embedding_legacy.go).
func legacyEmbeddingRequest(req *schemas.BifrostEmbeddingRequest) *schemas.BifrostEmbeddingRequest {
	if req == nil || req.Params == nil {
		return req
	}
	req.Input = lib.NormalizeLegacyEmbeddingInput(req.Input, req.Params.ExtraParams)
	return req
}
