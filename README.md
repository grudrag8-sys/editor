# Avatar swap (free, runs in the browser)

Tracks your body, hands and face in a video and moves a 3D avatar the same way. Everything runs on the user's device. No GPU server, no database, no uploads.

## Deploy on Cloudflare Pages (free)
1. Put these three files (index.html, app.js, README.md) at the root of your GitHub repo.
2. Cloudflare dashboard > Workers & Pages > Create > Pages > Connect to Git > pick the repo.
3. Framework preset: None. Build command: leave empty. Output directory: `/`.
4. Deploy. Open the link on your phone or computer (HTTPS is required and Pages provides it).

## Use
1. Choose a video (30 minutes maximum).
2. Optional: choose a .vrm avatar (free from VRoid Studio / VRoid Hub). Without one you get a mannequin.
3. Track motion (pass 1, can run at half speed on slow phones), then Preview or Render.
4. Output is the avatar on a green background (change colour in the app). Put it over your edit and key out the green.

## Known limits
- Not tested on real footage yet. If the avatar mirrors you, tick Mirror. If arms move toward or away from the camera the wrong way, tick Flip depth.
- Full 360 turns can confuse the tracker (left/right swap when you face away).
- Rendering plays the video in real time (30 minutes takes 30 minutes). Keep the tab open and screen on.
- Models load from jsDelivr and Google storage the first time (about 30 MB).
