# Signa — ASL Alphabet Recognition

Signa includes two versions of the ASL alphabet spelling project:

- **Desktop app:** PySide6 camera dashboard with local speech.
- **Web app:** Streamlit/WebRTC dashboard with browser-side speech.

Both use the project's existing PyTorch landmark classifier and MediaPipe
Tasks hand detector. The model recognizes A–Z, `space`, and `del`; it does
not include a `nothing` class or map letters directly to words.

## Desktop app

Create a separate environment for the desktop app, then install its
dependencies:

```bash
python -m venv .venv-desktop
# macOS/Linux
source .venv-desktop/bin/activate
# Windows PowerShell
# .venv-desktop\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
python webcam_landmark_v2.py
```

The desktop app uses PySide6 and `pyttsx3`. It opens the local webcam and
speaks completed words on the computer running the app.

## Web app — run locally

Python 3.12 is recommended for the web deployment dependency set.

```bash
python -m venv .venv-web
```

Activate the environment:

```bash
# macOS/Linux
source .venv-web/bin/activate

# Windows PowerShell
.venv-web\Scripts\Activate.ps1
```

Install dependencies and run the app from the repository root:

```bash
python -m pip install -r web_app/requirements.txt
python -m streamlit run web_app/app.py
```

Open the localhost URL printed by Streamlit, select **Start camera**, and
grant the browser camera permission.

## Recognition controls

- Hold a letter sign steady until it is accepted.
- Change to a different stable gesture to enter the next letter.
- To repeat the same letter, briefly remove the hand from the frame, then sign
  it again.
- `space` separates words and speaks the completed word using browser speech.
- `del` removes the previous character.
- Browser speech can be muted; completed words and the full sentence can be
  replayed.

## Deploy the web app on Streamlit Community Cloud

1. Create a Community Cloud app from this GitHub repository.
2. Select branch `main` and main file `web_app/app.py` (not
   `webcam_landmark_v2.py`, which launches the desktop application).
3. In **Advanced settings**, select Python 3.12. The pinned PyTorch version
   does not provide wheels for Python 3.14.
4. Deploy and open the resulting HTTPS URL.
5. Grant browser camera permission.

The deployment includes the Linux runtime libraries required by MediaPipe's
OpenCV contrib dependency.

The WebRTC configuration uses a public STUN server by default. Some firewalls
and hosted networks require TURN. If needed, add these environment variables
to the app's secrets/configuration:

```toml
[turn]
url = "turn:your-turn-host:3478"
username = "your-turn-username"
credential = "your-turn-credential"
```

Never commit TURN credentials. On restrictive networks, verify the hosted
camera connection over HTTPS; localhost testing does not verify public
WebRTC connectivity.

## Privacy and model assets

Video frames are processed in memory and are not saved by the application.
The repository includes `sign_landmark.pt` (about 219 KiB) and
`hand_landmarker.task` (about 7.5 MiB), for about 7.7 MiB total.

The desktop application is included in this repository. The web app is
deployed separately from `web_app/app.py`; deploying the web version does not
replace the desktop launcher.

## Tests

```bash
python -m unittest discover -s tests -v
node --test tests/test_browser_speech.js
```
