import mediapipe as mp

try:
    # Testando os módulos clássicos
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("\nVITÓRIA ABSOLUTA! O MediaPipe clássico está rodando perfeitamente!")
except Exception as e:
    print(f"\nERRO: {e}")