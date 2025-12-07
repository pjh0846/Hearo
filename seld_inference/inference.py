"""
inference.py

Real-time and file-based SELD inference with stereo audio.
Portable module that can be used in other projects.

Usage:
    python inference.py                     # Live microphone input
    python inference.py --file test.wav     # Test with file
    python inference.py --list-devices      # List audio devices

Requirements:
    pip install torch numpy librosa joblib pyaudio
"""

import os
import sys
import time
import argparse
import pickle
import threading
import queue
import numpy as np
import torch
import joblib

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local imports
from model import SELDModel
import inference_utils as utils

# SELD class names (DCASE2020 - 14 classes)
CLASS_NAMES = [
    'alarm', 'baby', 'crash', 'dog', 'engine', 'female_scream',
    'female_speech', 'fire', 'footsteps', 'knock', 'male_scream',
    'male_speech', 'phone', 'piano'
]


class RealtimeSELD:
    def __init__(self, weights_dir=None, device='cuda', slide_interval_ms=500):
        """
        Initialize SELD inference engine.
        
        Args:
            weights_dir: Path to weights directory. Default: ./weights relative to this script.
            device: 'cuda' or 'cpu'
            slide_interval_ms: Inference interval in milliseconds
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.slide_interval_ms = slide_interval_ms
        
        # Default to ./weights relative to this script
        if weights_dir is None:
            weights_dir = os.path.join(SCRIPT_DIR, 'weights')
        
        # Load config from checkpoint
        config_path = os.path.join(weights_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Loaded config from: {config_path}")
        
        # Audio parameters
        self.sr = self.params['sampling_rate']
        self.hop_length = int(self.sr * self.params['hop_length_s'])
        self.n_fft = 2 ** (2 * self.hop_length - 1).bit_length()
        self.window_samples = int(5.0 * self.sr)  # 5 seconds
        
        # Calculate input channels
        n_channels = 2  # stereo mel
        if self.params.get('ms'): n_channels += 2
        if self.params.get('gamma'): n_channels += 1
        if self.params.get('ipd'): n_channels += 2
        if self.params.get('iv'): n_channels += 1
        if self.params.get('slite'): n_channels += 1
        
        # Load model
        model_path = os.path.join(weights_dir, 'best_model.pth')
        in_feat_shape = (1, n_channels, 251, self.params['nb_mels'])
        self.model = SELDModel(self.params, in_feat_shape)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['seld_model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(weights_dir, 'scaler_dev.pkl')
        self.scalers = joblib.load(scaler_path)
        print(f"Scaler loaded: {scaler_path}")
        
        # Audio buffer
        self.audio_buffer = np.zeros((2, self.window_samples), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Results queue
        self.results_queue = queue.Queue()
        self.running = False
        
        print(f"\nConfiguration:")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  Window: 5 seconds ({self.window_samples} samples)")
        print(f"  Slide interval: {slide_interval_ms} ms")
        print(f"  Device: {self.device}")
    
    def extract_features(self, audio):
        """Extract features from audio buffer."""
        features = utils.extract_stereo_features(
            audio, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, win_length=self.n_fft,
            nb_mels=self.params['nb_mels'], max_freq=self.params['max_freq'],
            use_gamma=self.params.get('gamma', False), 
            use_ipd=self.params.get('ipd', False),
            use_iv=self.params.get('iv', False), 
            use_slite=self.params.get('slite', False), 
            use_ms=self.params.get('ms', False)
        )
        
        # Normalize
        for ch, scaler in enumerate(self.scalers):
            channel_data = features[ch].reshape(-1, features.shape[2])
            features[ch] = scaler.transform(channel_data).reshape(features.shape[1], features.shape[2])
        
        return features
    
    def infer(self, features):
        """Run model inference."""
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(feat_tensor)
        
        # Parse multi-ACCDOA output
        result = utils.get_multiaccdoa_labels(
            output, self.params['nb_classes'], 
            dnorm=self.params.get('dnorm', False), params=self.params
        )
        return result
    
    def parse_results(self, result, debug=False):
        """Parse model output to human-readable format."""
        sed0, _, doa0, dist0, _, sed1, _, doa1, dist1, _, sed2, _, doa2, dist2, _ = result
        
        detections = []
        for slot, (sed, doa, dist) in enumerate([(sed0, doa0, dist0), (sed1, doa1, dist1), (sed2, doa2, dist2)]):
            sed_np = sed.cpu().numpy()[0]
            doa_np = doa.cpu().numpy()[0]
            
            if debug and slot == 0:
                last_sed = sed_np[-1]
                print(f"\n[DEBUG] Slot 0 SED values (last frame):")
                for i, (name, val) in enumerate(zip(CLASS_NAMES, last_sed)):
                    if val:
                        print(f"  {name}: {val}")
                
                max_sed = sed_np.max(axis=0)
                active_classes = np.where(max_sed)[0]
                if len(active_classes) > 0:
                    print(f"[DEBUG] Active classes (any frame): {[CLASS_NAMES[i] for i in active_classes]}")
                else:
                    print("[DEBUG] No active classes in any frame")
            
            last_frame = -1
            for cls_idx in range(self.params['nb_classes']):
                if sed_np[last_frame, cls_idx]:
                    x = doa_np[last_frame, cls_idx]
                    y = doa_np[last_frame, cls_idx + self.params['nb_classes']]
                    azimuth = np.degrees(np.arctan2(y, x))
                    detections.append({
                        'class': CLASS_NAMES[cls_idx],
                        'class_id': cls_idx,
                        'azimuth': azimuth,
                        'slot': slot
                    })
        
        # Unify similar detections (same class within 15 degrees)
        detections = self._unify_detections(detections, thresh_deg=15)
        
        return detections
    
    def _unify_detections(self, detections, thresh_deg=15):
        """Merge detections of the same class within threshold degrees."""
        if not detections:
            return detections
        
        unified = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find similar detections
            similar_azimuths = [det1['azimuth']]
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j in used:
                    continue
                if det1['class_id'] == det2['class_id']:
                    # Check angular distance
                    diff = abs(det1['azimuth'] - det2['azimuth'])
                    if diff > 180:
                        diff = 360 - diff
                    if diff < thresh_deg:
                        similar_azimuths.append(det2['azimuth'])
                        used.add(j)
            
            # Average the azimuths
            avg_azimuth = np.mean(similar_azimuths)
            unified.append({
                'class': det1['class'],
                'class_id': det1['class_id'],
                'azimuth': avg_azimuth,
                'slot': det1['slot']
            })
        
        return unified
    
    def display_results(self, detections):
        """Display detection results in terminal."""
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        if not detections:
            sys.stdout.write('[No sound detected]')
        else:
            parts = []
            for det in detections:
                # 4 directions: front(-45~+45), right(+45~+135), back(>135 or <-135), left(-135~-45)
                az = det['azimuth']
                if -45 <= az <= 45:
                    direction = "↑"  # front
                elif 45 < az <= 135:
                    direction = "→"  # right
                elif -135 <= az < -45:
                    direction = "←"  # left
                else:
                    direction = "↓"  # back
                parts.append(f"{det['class']}:{det['azimuth']:+.0f}°{direction}")
            sys.stdout.write(' | '.join(parts))
        
        sys.stdout.flush()
    
    def process_loop(self):
        """Main processing loop."""
        print("\n[Processing started - Press Ctrl+C to stop]")
        
        while self.running:
            start_time = time.time()
            
            with self.buffer_lock:
                audio = self.audio_buffer.copy()
            
            try:
                features = self.extract_features(audio)
                result = self.infer(features)
                detections = self.parse_results(result)
                self.display_results(detections)
            except Exception as e:
                print(f"\nError: {e}")
            
            elapsed = time.time() - start_time
            sleep_time = (self.slide_interval_ms / 1000.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for microphone input."""
        import pyaudio
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        audio_data = audio_data.reshape(-1, 2).T
        
        with self.buffer_lock:
            self.audio_buffer = np.roll(self.audio_buffer, -frame_count, axis=1)
            self.audio_buffer[:, -frame_count:] = audio_data
        
        return (None, pyaudio.paContinue)
    
    def run_microphone(self, device_index=None):
        """Run real-time inference with microphone input."""
        try:
            import pyaudio
        except ImportError:
            print("Error: PyAudio not installed. Run: pip install pyaudio")
            return
        
        p = pyaudio.PyAudio()
        
        chunk_size = int(self.sr * 0.1)
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sr,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.running = True
        stream.start_stream()
        
        try:
            self.process_loop()
        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def run_file(self, audio_path):
        """Run inference on audio file."""
        print(f"\nProcessing file: {audio_path}")
        
        audio, sr = utils.load_audio(audio_path, self.sr)
        if audio.shape[0] != 2:
            print(f"Error: Stereo audio required, got {audio.shape[0]} channels")
            return
        
        total_samples = audio.shape[1]
        slide_samples = int(self.sr * self.slide_interval_ms / 1000)
        
        print(f"Audio duration: {total_samples / self.sr:.1f}s")
        print("\n[Press Ctrl+C to stop]\n")
        
        self.running = True
        pos = 0
        
        try:
            while self.running and pos + self.window_samples <= total_samples:
                window = audio[:, pos:pos + self.window_samples]
                
                features = self.extract_features(window)
                result = self.infer(features)
                detections = self.parse_results(result)
                
                timestamp = pos / self.sr
                sys.stdout.write(f'\r[{timestamp:6.1f}s] ')
                self.display_results(detections)
                
                pos += slide_samples
                time.sleep(self.slide_interval_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")
        
        print("\n[Done]")


def list_audio_devices():
    """List available audio input devices."""
    try:
        import pyaudio
    except ImportError:
        print("Error: PyAudio not installed. Run: pip install pyaudio")
        return
    
    p = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] >= 2:
            print(f"  [{i}] {info['name']}")
            print(f"      Channels: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])} Hz")
    
    p.terminate()


def main():
    parser = argparse.ArgumentParser(description='SELD Inference (File or Realtime)')
    parser.add_argument('--file', type=str, help='Audio file to process (instead of microphone)')
    parser.add_argument('--device', type=int, default=None, help='Audio device index')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    parser.add_argument('--slide-ms', type=int, default=200, help='Slide interval in ms (default: 200)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights directory (default: ./weights)')
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Initialize
    seld = RealtimeSELD(weights_dir=args.weights, slide_interval_ms=args.slide_ms)
    
    if args.file:
        seld.run_file(args.file)
    else:
        seld.run_microphone(device_index=args.device)


if __name__ == '__main__':
    main()
