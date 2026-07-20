from alignment.arcface_aligner import ArcFaceAligner

def test_alignment_latency(benchmark, sample_image, sample_landmarks):
    aligner = ArcFaceAligner()
    
    # Run the benchmark
    aligned_face = benchmark(aligner.align, sample_image, sample_landmarks)
    
    # USE the variable so Ruff is happy (and to verify correctness)
    assert aligned_face is not None
    assert aligned_face.image.shape == (112, 112, 3)
    