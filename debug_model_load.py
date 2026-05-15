"""Debug script untuk test load model."""
import os
import sys
import traceback

# Add project dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Import custom objects dari apps
from apps import _make_custom_objects

# Test 1: Load model_b3.h5
print("\n" + "="*60)
print("Test 1: Loading model_b3.h5 (MobileNetV2)")
print("="*60)
try:
    model_v2 = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_b3.h5"),
        custom_objects=_make_custom_objects(),
        compile=False,
    )
    print("✓ Berhasil load model_b3.h5")
    print(f"  Input shape: {model_v2.input_shape}")
    print(f"  Output shape: {model_v2.output_shape}")
except Exception as e:
    print(f"✗ Gagal load model_b3.h5:")
    print(f"  Error: {str(e)[:200]}")
    traceback.print_exc()

# Test 2: Load model_b3.keras
print("\n" + "="*60)
print("Test 2: Loading model_b3.keras (MobileNetV2)")
print("="*60)
try:
    model_v2_keras = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_b3.keras"),
        custom_objects=_make_custom_objects(),
        compile=False,
    )
    print("✓ Berhasil load model_b3.keras")
    print(f"  Input shape: {model_v2_keras.input_shape}")
    print(f"  Output shape: {model_v2_keras.output_shape}")
except Exception as e:
    print(f"✗ Gagal load model_b3.keras:")
    print(f"  Error: {str(e)[:200]}")
    traceback.print_exc()

# Test 3: Load model_b3_v3.h5
print("\n" + "="*60)
print("Test 3: Loading model_b3_v3.h5 (MobileNetV3)")
print("="*60)
try:
    model_v3 = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_b3_v3.h5"),
        custom_objects=_make_custom_objects(),
        compile=False,
    )
    print("✓ Berhasil load model_b3_v3.h5")
    print(f"  Input shape: {model_v3.input_shape}")
    print(f"  Output shape: {model_v3.output_shape}")
except Exception as e:
    print(f"✗ Gagal load model_b3_v3.h5:")
    print(f"  Error: {str(e)[:200]}")
    traceback.print_exc()

# Test 4: Load model_b3_v3.keras
print("\n" + "="*60)
print("Test 4: Loading model_b3_v3.keras (MobileNetV3)")
print("="*60)
try:
    model_v3_keras = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_b3_v3.keras"),
        custom_objects=_make_custom_objects(),
        compile=False,
    )
    print("✓ Berhasil load model_b3_v3.keras")
    print(f"  Input shape: {model_v3_keras.input_shape}")
    print(f"  Output shape: {model_v3_keras.output_shape}")
except Exception as e:
    print(f"✗ Gagal load model_b3_v3.keras:")
    print(f"  Error: {str(e)[:200]}")
    traceback.print_exc()

print("\n" + "="*60)
print("Debug Summary")
print("="*60)
print("Selesai! Cek output di atas untuk info model loading.")

