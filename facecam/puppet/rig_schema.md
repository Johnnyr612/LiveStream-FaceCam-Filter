# Puppet Rig Schema (MVP)

Each puppet pack lives in:
assets/puppets/<puppet_name>/

Required files:
- rig.json
- layer PNGs (with transparency)

rig.json format:
{
  "name": "plush_01",
  "base_scale": 1.0,
  "canvas_size": [512, 512],
  "pivot": [256, 280],
  "layers": {
    "body": "body.png",
    "eyes_open": "eyes_open.png",
    "eyes_closed": "eyes_closed.png",
    "mouth_closed": "mouth_closed.png",
    "mouth_open": "mouth_open.png"
  }
}

Notes:
- All PNG layers must be the same canvas size.
- pivot is where the puppet “attaches” to the face (we place it near center/neck area).
