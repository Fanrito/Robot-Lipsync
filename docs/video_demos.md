# README Video Demos

The homepage embeds three native GitHub video players: natural speech, bilabial speech and
a collapsible controlled-motion ablation. They contain the **same muted H.264 bytes** already
published under `assets/videos/`, not new raw recordings. They illustrate motion only; no audio
listening or physical articulation-improvement claim is made.

The [original media manifest](../assets/media_manifest.json) records source hashes, duration,
column order and scope. [Attachment bindings](../assets/video_attachments.json) associate those
files with GitHub-hosted URLs. [Issue #1](https://github.com/Fanrito/Robot-Lipsync/issues/1)
is a stable media archive, not a bug report.

## Why Attachments

GitHub's official [attachment CLI guide](https://docs.github.com/en/github-cli/github-cli/attaching-files-with-github-cli)
documents native video uploads using `--attach`. The uploaded URL appears alone in a Markdown
paragraph and is rendered as a video player. A repository-relative MP4 link remains useful as
a fallback but is not the same as an inline attachment player.

The videos were uploaded using the existing CLI, without an environment upgrade:

```bash
gh issue create --repo Fanrito/Robot-Lipsync \
  --title "README demo videos (public media archive)" \
  --body-file path/to/media_archive_body.md \
  --attach assets/videos/speech_natural.mp4 \
  --attach assets/videos/speech_bilabial.mp4 \
  --attach assets/videos/controlled.mp4
```

This documents the initial upload; **do not rerun it for every README edit**. Reuse the existing
attachment URLs unless the media itself changes. If future videos are uploaded, review rights
and privacy first, keep source hashes and captions, and preserve the archival issue. Do not
substitute raw robot recordings or restore lab audio without a separate release decision.

## Playback

Click Play in the README; expand the controlled-motion section for the third clip. GitHub may
pause/hide motion according to browser or accessibility settings. All clips are intentionally
silent. If the player fails to load, use the repository MP4 link below it, or open the tracked
file locally. Native attachments and repository copies are independent hosting paths.

The pipeline figures and metric tables are unchanged. No new models, inference, robot execution
or acquisition is involved in this presentation update.
