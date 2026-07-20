from alignment.arcface_aligner import ArcFaceAligner

def test_alignment_latency(benchmark, sample_image, sample_landmarks):
    aligner = ArcFaceAligner()
    
    # Benchmark the alignment process
    aligned_face = benchmark(aligner.align, sample_image, sample_landmarks)
    
    # Assert it completed within an acceptable timeframe (e.g., < 5ms)
    assert benchmark.stats.stats.mean < 0.005
    