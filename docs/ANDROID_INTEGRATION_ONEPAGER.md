# Miyraa NLP ↔ Android Integration (One-Pager)

**Version:** 1.0  
**Last Updated:** December 18, 2025  
**Audience:** Android feature teams preparing demo-ready builds

---

## 1. Drop-In SDK Setup
- Add the SDK module (see `sdk/android/README.md`) and initialize `MiyraaSDK` once per Application.
- Configure the base URL from remote config; default staging endpoint is `https://staging.api.miyraa.ai`.
- Enable structured logging with the shared `MiyraaLogger` so privacy filters match the backend pipeline.

## 2. Request → UI Flow
1. Call `sdk.analyze(text)` (or `analyzeBatch`) on a background dispatcher.
2. Pass every `EmotionResult` to `NlpUiMapper.fromResult(result)` in `sdk/android/NlpUiMapping.kt`.
3. Render UI strictly from the returned `UiPresentation` (color palette, warmth, motion, tone).
4. If the SDK throws or returns null, call `NlpUiMapper.fromResult(null)` for a safe fallback.

> Reference: [ANDROID_UI_MAPPING.md](ANDROID_UI_MAPPING.md)

## 3. Safety & Privacy Defaults
- Backend requests scrub PII automatically; no raw text is logged (see [docs/API.md](API.md)).
- UI must hide emotion widgets when `UiPresentation.isSafeToRender == false` and show the neutral warning card.
- Respect `UiPresentation.klynAiTone` to keep assistant phrasing aligned with compliance commitments.

## 4. Telemetry & Analytics Hooks
- Emit `MiyraaEvent.EmotionRendered` with emotion, warmth bucket, and safety flag (never include raw text).
- Emit `MiyraaEvent.NlpFallback` when `fallbackReason != null` to monitor availability.
- Use the shared `analytics-miyraa` schema version `2.1` so dashboards stay consistent.

## 5. Release Checklist (Demo Readiness)
- [ ] Emotion chips reflect the palette for all 11 labels.
- [ ] Motion speeds match warmth buckets (FAST 140 ms / MEDIUM 220 ms / SLOW 360 ms).
- [ ] Safety curtain appears when `isSafeToRender` is false.
- [ ] Offline mode shows retry CTA within 500 ms.
- [ ] KlynAI voice honors `klynAiTone` without custom branching.
- [ ] Privacy audit confirms no PII in logs or analytics payloads.

## 6. Support Contacts
- **Android Lead:** maya@klynai.com
- **NLP Platform:** platform@miyraa.ai
- **Incident Hotline:** #miyraa-android-alerts (Slack)
