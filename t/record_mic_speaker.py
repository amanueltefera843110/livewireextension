# record_mic_speaker.py
# Unlimited recording of MIC + SYSTEM (Voicemeeter/VB-Cable). Press ENTER once to stop.
# Now with OpenAI Whisper transcription support!
import argparse, sys, time, threading, queue, os
import numpy as np
import sounddevice as sd
import soundfile as sf
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ===== Defaults (can be overridden by CLI) =====
DEFAULT_SR_CANDIDATES = (48000, 44100)
BLOCKSIZE = 1024
DTYPE = 'float32'
LATENCY = 'high'
MIC_WAV = "mic.wav"
SYS_WAV = "system.wav"
MERGED_WAV = "merged.wav"
TRANSCRIPT_TXT = "transcript.txt"
# ==============================================

def list_devices():
    print("[INFO] Devices:")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:>3}: {d['name']} | in {d['max_input_channels']} / out {d['max_output_channels']} | hostapi {d['hostapi']}")

def probe_samplerate(dev_idx: int, ch: int, sr_candidates=DEFAULT_SR_CANDIDATES) -> int:
    for sr in sr_candidates:
        try:
            with sd.InputStream(device=dev_idx, samplerate=sr, channels=ch):
                return sr
        except Exception:
            continue
    raise RuntimeError(f"Device {dev_idx} supports neither {sr_candidates} for {ch}ch.")

def writer_thread(path, q: "queue.Queue[np.ndarray]", stop_evt: threading.Event, sr: int, ch: int):
    with sf.SoundFile(path, mode='w', samplerate=sr, channels=ch, subtype='PCM_16') as f:
        while not (stop_evt.is_set() and q.empty()):
            try:
                f.write(q.get(timeout=0.1))
            except queue.Empty:
                pass

def record_stream(dev_idx: int, sr: int, ch: int, label: str, out_path: str,
                  stop_evt: threading.Event, vu_state: dict, blocksize=BLOCKSIZE,
                  dtype=DTYPE, latency=LATENCY):
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=60)
    wt = threading.Thread(target=writer_thread, args=(out_path, q, stop_evt, sr, ch), daemon=True)
    wt.start()

    def cb(indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"[{label}] {status}\n")
        q.put(indata.copy())
        # Simple RMS meter
        vu_state[label] = float(np.sqrt(np.mean(indata.astype(np.float32)**2)))

    with sd.InputStream(device=dev_idx, samplerate=sr, channels=ch,
                        dtype=dtype, blocksize=blocksize, latency=latency,
                        callback=cb):
        print(f"[{label}] dev={dev_idx}  sr={sr}  ch={ch}  (recording)")
        while not stop_evt.is_set():
            sd.sleep(50)
    wt.join(timeout=2)

def merge_files(mic_path: str, sys_path: str, out_path: str, sr: int):
    try:
        mic, _  = sf.read(mic_path, dtype='float32', always_2d=True)
        sysm, _ = sf.read(sys_path, dtype='float32', always_2d=True)
    except Exception as e:
        print("[WARN] Merge skipped (missing/invalid input files):", e)
        return

    # Pad to equal length
    n = max(len(mic), len(sysm))
    if len(mic)  < n: mic  = np.pad(mic,  ((0, n - len(mic)),  (0, 0)))
    if len(sysm) < n: sysm = np.pad(sysm, ((0, n - len(sysm)), (0, 0)))

    # Ensure both are stereo before summing
    if mic.shape[1]  == 1: mic  = np.repeat(mic,  2, axis=1)
    if sysm.shape[1] == 1: sysm = np.repeat(sysm, 2, axis=1)

    mix = mic + sysm
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.99:
        mix = mix / peak * 0.95

    sf.write(out_path, mix, sr, subtype='PCM_16')
    print(f"[OK] merged → {out_path}")

def transcribe_audio(audio_path: str, api_key: str = None, model: str = "whisper-1", language: str = None) -> str:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_path: Path to audio file
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        model: Model to use (default: whisper-1)
        language: Optional language code (e.g., 'en', 'es', 'fr')
    
    Returns:
        Transcribed text
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library not installed. Install with: pip install openai")
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    print(f"[TRANSCRIBE] Processing {audio_path}...")
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                response_format="text"
            )
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

def save_transcript(text: str, output_path: str = TRANSCRIPT_TXT):
    """Save transcript to text file"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] transcript saved → {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Record MIC + SYSTEM (Voicemeeter/VB-Cable). Press ENTER to stop.")
    ap.add_argument("--list", action="store_true", help="List devices and exit")
    ap.add_argument("--mic", type=int, help="Mic input device index (from --list)")
    ap.add_argument("--sys", type=int, help="System input device index (Voicemeeter Out B1 or CABLE Output)")
    ap.add_argument("--sr", type=int, choices=[44100, 48000], help="Force sample rate (otherwise auto-probe)")
    ap.add_argument("--mic-ch", type=int, help="Force mic channels (default: auto 1 if available)")
    ap.add_argument("--sys-ch", type=int, help="Force system channels (default: auto 2 if available, else 1)")
    ap.add_argument("--transcribe", action="store_true", help="Transcribe the merged audio using OpenAI Whisper")
    ap.add_argument("--transcribe-mic", action="store_true", help="Transcribe only the mic audio")
    ap.add_argument("--transcribe-sys", action="store_true", help="Transcribe only the system audio")
    ap.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    ap.add_argument("--language", type=str, help="Language code for transcription (e.g., 'en', 'es', 'fr')")
    ap.add_argument("--mic-only", action="store_true", help="Record only microphone (skip system audio)")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    # Resolve defaults from sd.default if not provided
    indev, outdev = sd.default.device
    mic_dev = args.mic if args.mic is not None else indev
    
    # Only resolve system device if not in mic-only mode
    if args.mic_only:
        sys_dev = None
        sys_info = None
    else:
        sys_dev = args.sys if args.sys is not None else outdev

    if mic_dev is None:
        list_devices()
        print("\n[ERR] Could not resolve default mic device. Re-run with --mic <idx>.")
        return
    
    if not args.mic_only and sys_dev is None:
        list_devices()
        print("\n[ERR] Could not resolve default system device.")
        print("      Options:")
        print("      1. Use --mic-only to record only microphone")
        print("      2. Use --sys <idx> to specify a system input device (see --list)")
        return

    mic_info = sd.query_devices(mic_dev)
    if not args.mic_only:
        sys_info = sd.query_devices(sys_dev)

    # Channel counts: prefer mic=1ch, system=2ch when available
    mic_ch = args.mic_ch if args.mic_ch else (1 if mic_info['max_input_channels'] >= 1 else mic_info['max_input_channels'])
    if mic_ch < 1:
        print(f"[ERR] Mic device {mic_dev} has no input channels.")
        return

    if not args.mic_only:
        sys_ch = args.sys_ch if args.sys_ch else (2 if sys_info['max_input_channels'] >= 2 else (1 if sys_info['max_input_channels'] >= 1 else 0))
        if sys_ch < 1:
            list_devices()
            print(f"\n[ERR] System device {sys_dev} ({sys_info['name']}) has no input channels.")
            print("      Options:")
            print("      1. Use --mic-only to record only microphone")
            print("      2. Use --sys <idx> to specify a device with input channels (see list above)")
            print("      3. Install a virtual audio cable (like VB-Cable or BlackHole) for system audio capture")
            return

    # Sample rate
    if args.sr:
        sr = args.sr
        # quick validation
        try:
            with sd.InputStream(device=mic_dev, samplerate=sr, channels=mic_ch): pass
            if not args.mic_only:
                with sd.InputStream(device=sys_dev, samplerate=sr, channels=sys_ch): pass
        except Exception as e:
            print(f"[ERR] Forced SR {sr} not supported by one of the devices:", e)
            return
    else:
        # Probe mic; then try to keep both at the same SR; if sys fails, fallback to 44.1k
        sr_m = probe_samplerate(mic_dev, mic_ch)
        if not args.mic_only:
            try:
                with sd.InputStream(device=sys_dev, samplerate=sr_m, channels=sys_ch): sr = sr_m
            except Exception:
                sr = 44100
                # Re-probe mic at 44.1k if needed
                probe_samplerate(mic_dev, mic_ch, sr_candidates=(44100,))
        else:
            sr = sr_m

    print(f"[INFO] MIC: idx={mic_dev}  name={mic_info['name']}  in={mic_info['max_input_channels']}")
    if not args.mic_only:
        print(f"[INFO] SYS: idx={sys_dev}  name={sys_info['name']}  in={sys_info['max_input_channels']}")
    else:
        print(f"[INFO] MODE: Mic-only recording (system audio disabled)")
    print(f"[INFO] SR={sr}  BLOCK={BLOCKSIZE}  DTYPE={DTYPE}")
    print("\n[INFO] 🔴 Recording… Press ENTER once to stop.\n")

    stop_evt = threading.Event()
    vu = {"MIC": 0.0, "SYSTEM": 0.0}

    t_mic = threading.Thread(target=record_stream, args=(mic_dev, sr, mic_ch, "MIC", MIC_WAV, stop_evt, vu), daemon=True)
    t_mic.start()
    
    if not args.mic_only:
        t_sys = threading.Thread(target=record_stream, args=(sys_dev, sr, sys_ch, "SYSTEM", SYS_WAV, stop_evt, vu), daemon=True)
        t_sys.start()
    else:
        t_sys = None

    # Input detection in separate thread
    def wait_for_enter():
        """Wait for ENTER key press in a separate thread"""
        try:
            input()  # This blocks until ENTER is pressed
            stop_evt.set()
        except (EOFError, KeyboardInterrupt):
            stop_evt.set()
    
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()
    
    # Simple live VU (optional)
    print("\n[INFO] Press ENTER to stop recording...\n")
    try:
        while not stop_evt.is_set():
            if args.mic_only:
                print(f"[VU] MIC={vu['MIC']:.5f}      ", end="\r", flush=True)
            else:
                print(f"[VU] MIC={vu['MIC']:.5f}  SYS={vu['SYSTEM']:.5f}      ", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_evt.set()
    finally:
        # Clear VU line
        print(" " * 60, end="\r", flush=True)
        print("\n[INFO] Stopping recording...")
        # Wait for threads to finish
        t_mic.join(timeout=3)
        if t_sys:
            t_sys.join(timeout=3)
        input_thread.join(timeout=0.5)

    if args.mic_only:
        print(f"[DONE] saved: {MIC_WAV}")
        # For mic-only, the "merged" file is just the mic
        import shutil
        shutil.copy(MIC_WAV, MERGED_WAV)
        print(f"[DONE] copied mic to: {MERGED_WAV}")
    else:
        merge_files(MIC_WAV, SYS_WAV, MERGED_WAV, sr)
        print(f"[DONE] saved: {MIC_WAV}  {SYS_WAV}  {MERGED_WAV}")
    
    # Transcription
    if args.transcribe or args.transcribe_mic or args.transcribe_sys:
        if not OPENAI_AVAILABLE:
            print("\n[WARN] OpenAI library not available. Install with: pip install openai")
            print("       Skipping transcription.")
        else:
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n[WARN] OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api-key")
                print("       Skipping transcription.")
            else:
                print("\n[INFO] Starting transcription...")
                try:
                    if args.transcribe:
                        # Transcribe merged audio
                        transcript = transcribe_audio(MERGED_WAV, api_key=api_key, language=args.language)
                        save_transcript(transcript, "transcript_merged.txt")
                        print(f"\n[TRANSCRIPT]\n{transcript}\n")
                    
                    if args.transcribe_mic:
                        # Transcribe mic audio
                        transcript = transcribe_audio(MIC_WAV, api_key=api_key, language=args.language)
                        save_transcript(transcript, "transcript_mic.txt")
                        print(f"\n[MIC TRANSCRIPT]\n{transcript}\n")
                    
                    if args.transcribe_sys:
                        # Transcribe system audio
                        transcript = transcribe_audio(SYS_WAV, api_key=api_key, language=args.language)
                        save_transcript(transcript, "transcript_sys.txt")
                        print(f"\n[SYSTEM TRANSCRIPT]\n{transcript}\n")
                        
                except Exception as e:
                    print(f"\n[ERR] Transcription failed: {e}")

if __name__ == "__main__":
    main()
