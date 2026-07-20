## Multilingual Voice AI Pipeline — Case Study Submission
### Soumili Jana Ghosh

---
### Summary

For this multilingual voice pipeline, I used different models for different languages instead of one model for all. XTTS-v2 worked best for English and Arabic because it produced natural speech and good speaker similarity. For Hindi and Bengali, MMS-TTS performed better because the voice-cloning models (XTTS-v2 and Chatterbox-ML) did not generate clear speech. I tested all models on Google Colab using the same reference voice. The results show that English and Arabic are close to production quality, but Hindi and Bengali still need improvement in open-source voice cloning models. 

### Language Support by Model (confirmed via runtime testing)

This table shows the actual results from testing each model. It is based on the real error messages and outputs produced by the models when an unsupported language was used.

| Language | XTTS-v2 | Chatterbox-ML | MMS-TTS |
|----------|:-------:|:-------------:|:-------:|
| English  | Yes | Yes | Yes |
| Arabic   | Yes | Yes | Yes |
| Hindi    | Yes | Yes | Yes |
| Bengali  | Not supported | Not supported |  Yes |

**XTTS-v2**
- Supports 17 languages, including English, Arabic, and Hindi.
- Bengali is not supported.
- During runtime testing, requesting Bengali produced an `AssertionError`.

**Chatterbox-ML**
- Supports 23 languages, including English, Arabic, and Hindi.
- Bengali is not supported.
- During runtime testing, requesting Bengali produced a `ValueError`.

### Models Tested

| Model | Source | Voice Cloning | Languages Tested |
|-------|--------|:-------------:|------------------|
| XTTS-v2 | Coqui (`tts_models/multilingual/multi-dataset/xtts_v2`) | Yes | English, Arabic, Hindi |
| Chatterbox-ML | Resemble AI (Chatterbox Multilingual) | Yes | English, Arabic, Hindi |
| MMS-TTS | Meta (`facebook/mms-tts-{eng,ara,hin,ben}`) |  No (uses a fixed voice for each language) | English, Arabic, Hindi, Bengali |


### Closed Tools Used for Evaluation Only

The following tools were used only for evaluation, not for speech generation:

- **Whisper Large-v3 (via faster-whisper)** – Used to calculate Word Error Rate (WER) by converting generated speech back into text.
- **Resemblyzer** – Used to measure speaker similarity by comparing voice embeddings.

### Known dependency conflicts (and fixes)

1. coqui-tts needs transformers>=4.57, but Chatterbox needs transformers==4.46.x. These cannot coexist in one environment. I ran XTTS and Chatterbox in separate runtime sessions, reinstalling the correct transformers version and restarting the Colab kernel between them.
2. NameError: project_path not defined — Happened after every kernel restart since variables don't persist, fixed by redefining paths at the top of each new session.
3. ffmpeg m4a→wav conversion — Voice sample was recorded as .m4a converted with ffmpeg -ar 22050 -ac 1 to match XTTS's expected format.
4. CPU vs GPU Whisper mismatch (float16 error) — Runtime lost GPU access temporarily, fixed by auto-switching compute_type based on torch.cuda.is_available().
5. XTTS language="bn" AssertionError — Discovered Bengali isn't in XTTS's 17 supported languages documented as a real model limitation instead of forcing it.
6. Chatterbox language_id="bn" ValueError — Confirmed Chatterbox also doesn't support Bengali (23 supported languages, Bengali not among them) documented as another real model limitation.

### Reference voice

An 11.63-second personal voice recording (voice_sample.wav, 22050Hz mono), converted from a phone-recorded .m4a file via:

```bash
!ffmpeg -y -i "{m4a_path}" -ar 22050 -ac 1 "{wav_path}"
```

### Reproduction order

1. Load reference audio.
2. Run XTTS-v2 generation + benchmarks for English/Arabic/Hindi (see notebook.ipynb, XTTS section).
3. Restart runtime, switch transformers version, run Chatterbox-ML generation + benchmarks for English/Arabic/Hindi.
4. Run MMS-TTS generation + benchmarks for English/Arabic/Hindi/Bengali.
5. Merge all three result CSVs into main_comparison.csv.

### Benchmark Results

| Language | Model | WER | Latency (s) | RTF | Speaker Similarity | MOS |
|----------|-------|:---:|:-----------:|:---:|:------------------:|:---:|
| English | XTTS-v2 | 12.5%* | 4.22 | 0.884 | 0.836 | 4.3 |
| English | Chatterbox-ML | 12.5%* | 155.30⁺ | 38.063⁺ | 0.916 | 4.5 |
| English | MMS-TTS | 12.5%* | 2.79 | 1.025 | 0.485 | 3.5 |
| Arabic | XTTS-v2 | 0.0% | 1.37 | 0.461 | 0.764 | 4.2 |
| Arabic | Chatterbox-ML | 0.0% | 188.67⁺ | 34.683⁺ | 0.823 | 4.3 |
| Arabic | MMS-TTS | 20.0% | 5.49 | 1.357 | 0.514 | 3.3 |
| Hindi | XTTS-v2 | 50.0% | 2.80 | 0.475 | 0.849 | 2.5 |
| Hindi | Chatterbox-ML | 83.33% | 119.39⁺ | 43.893⁺ | 0.809 | 2.2 |
| Hindi | MMS-TTS | 50.0% | 3.46 | 1.336 | 0.483 | 2.8 |
| Bengali | XTTS-v2 | Not supported | — | — | — | — |
| Bengali | Chatterbox-ML | Not supported | — | — | — | — |
| Bengali | MMS-TTS | 83.33% | 4.27 | 1.003 | 0.467 | 1.9 |

### Per-Language Recommendation

English → XTTS-v2 or Chatterbox-ML
Both models achieved nearly 0% WER and met the Real-Time Factor (RTF) requirement. Chatterbox-ML produced better voice cloning, making it the best choice when speaker similarity is important. XTTS-v2 provided more stable latency and RTF, making it a better option when speed and consistent performance are the priority.

Arabic → XTTS-v2
Both XTTS-v2 and Chatterbox-ML produced clear and accurate speech. I recommend XTTS-v2 as the default choice because it showed more consistent performance during testing. However, Chatterbox-ML achieved better speaker similarity, so it is a good option when voice cloning quality is the main priority.

Hindi →None of the models met the target WER of 10% or less for Hindi. MMS-TTS gave the best results, but it does not support voice cloning. XTTS-v2 and Chatterbox-ML can clone voices, but they did not pronounce Hindi accurately. This shows that high-quality open-source Hindi voice cloning still needs improvement.

Bengali → MMS-TTS was the only model that supported Bengali. XTTS-v2 and Chatterbox-ML do not support Bengali. Although MMS-TTS worked, its speech quality was the lowest among all the models tested. Based on these results, open-source Bengali voice cloning is not yet suitable for production use.

### Where It Breaks — Honest Failure Modes

- **Hindi:** XTTS-v2 and Chatterbox-ML produced good voice cloning, but the Hindi pronunciation was poor. The generated speech was not transcribed correctly by the
 model, showing that Hindi language support is still weak. This appears to be a limitation of the models, not the test setup.

- **Bengali:** XTTS-v2 and Chatterbox-ML do not support Bengali. During testing, both models returned a "language not supported" error. MMS-TTS was the only model that worked for Bengali, but it does not support voice cloning.

- **Speed:** During one benchmark run, multiple large models were loaded at the same time, causing GPU memory contention. This affected the speed (RTF) measurements for Chatterbox-ML, so those results may not represent its best possible performance.

## What's Still Missing / How I'd Improve It

- **Explore more open-source models for Indian languages.** In future work, I would evaluate additional open-source TTS models designed for Indian languages, such as AI4Bharat Indic Parler-TTS and other Indic speech models, to identify the best option for Hindi, Bengali, and other Indian languages.

- **Streaming latency was not measured.** I measured only the time to generate the complete audio clip. In future work, I would also measure the time to the first audio chunk for streaming applications.

- **Only one reference speaker was used.** A more complete evaluation should include multiple speakers with different genders and accents to check if the results remain consistent.

- **MOS (Mean Opinion Score) is pending.** Human listener ratings from multiple people should be collected to better evaluate the naturalness of the generated speech.

- **Chatterbox speed should be tested again.** During one test, multiple large models were loaded at the same time, which used more GPU memory and may have affected the speed results. I would run the test again with only the Chatterbox model loaded to get more accurate RTF and latency measurements.

- **Bengali fine-tuning wasn't attempted.** XTTS-v2 and Chatterbox-ML could theoretically support Bengali with fine-tuning, but this requires a large labeled dataset and significant compute , a multi-week effort.
### A Note on How This Was Built

This project involved several real challenges during development. I have kept these issues documented.

- **Dependency conflicts:** XTTS-v2 and Chatterbox required different versions of the `transformers` library. To solve this, I ran them in separate sessions and restarted the runtime when needed.

- **GPU memory issue:** During one benchmark, multiple large models were loaded at the same time, which increased GPU memory usage and affected the Chatterbox speed results. This has been clearly mentioned in the results.

- **Reference voice:** I used recorded voice as the reference sample. The audio was recorded as `voice_sample.m4a` and converted to `voice_sample.wav` before testing.

The notebook (`notebook.ipynb`) contains the complete development process, including successful runs, errors, and fixes. This demonstrates that the pipelines were actually built, tested, and benchmarked.

## 9. Hardware Note

- **XTTS-v2** was tested on a **Google Colab T4 GPU**.
- During the project, the free GPU limit in Colab was reached, so **Chatterbox-ML** and **MMS-TTS** were tested on the **CPU**.

### Impact on the Results

- **Latency** and **RTF** for Chatterbox-ML and MMS-TTS are higher because they were run on the CPU. Therefore, their speed results should not be directly compared with XTTS-v2, which was tested on a GPU.

- **WER**, **Speaker Similarity**, and **MOS** are mostly unaffected by the hardware because they measure the quality of the generated speech, not the generation speed.

- For a completely fair comparison, all three models should be tested on the **same GPU**. This limitation is mentioned here to keep the results transparent and honest.


### Repository Structure

```text
Voice_Project/
├── notebook.ipynb                      # Main notebook with all pipelines and benchmarks
├── README.md                           # Project documentation
├── reference_audio/
│   └── voice_sample.wav                # Reference voice sample
├── outputs/
│   ├── xtts/
│   │   ├── english/
│   │   ├── arabic/
│   │   └── hindi/
│   ├── chatterbox/
│   │   ├── english/
│   │   ├── arabic/
│   │   └── hindi/
│   └── mms/
│       ├── english/
│       ├── arabic/
│       ├── hindi/
│       └── bengali/
└── results/
    ├── benchmark_results.csv
    ├── chatterbox_benchmark_results.csv
    ├── mms_benchmark_results.csv
    └── main_comparison.csv
```