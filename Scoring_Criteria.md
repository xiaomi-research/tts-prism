# 12-Dimensional Scoring Criteria for TTS-PRISM

This document details the explicit, fine-grained quantitative rubrics used for the schema-driven instruction tuning of TTS-PRISM. The evaluation schema is hierarchically divided into a **Basic Capability Layer** (Scores 1-5) and an **Advanced Expressiveness Layer** (Scores 0-2).

---

## Part I: Basic Capability Layer (Score 1-5)
This layer assesses the fundamental robustness, intelligibility, and stability of the synthesized speech. 

### 1. Pronunciation Accuracy
**Definition:** Evaluates strict adherence to standard Mandarin pronunciation (e.g., based on the Contemporary Chinese Dictionary). This includes accurate polyphone disambiguation based on semantic context, proper tone sandhi, and precise retroflex/flat or nasal/lateral consonant differentiation. It also assesses the accurate reading of numbers, dates, URLs, emails, foreign loanwords, code-switching fluency, and specific tonal patterns/rhymes (平仄/韵脚) in ancient poetry.
* **Score 5:** Flawless pronunciation. Clear, fluent, and natural articulation without any dropped or added phonemes. Code-switching, numbers, and special symbols are pronounced accurately and naturally.
* **Score 4:** Mostly accurate with only minor deviations. These slight errors (e.g., slight accent, slightly imperfect or muffled articulation) do not impair comprehension. No dropped or added syllables.
* **Score 3:** A few key pronunciation errors or inconsistent readings (e.g., 1-2 instances of incorrect tones or confused nasals/laterals) causing slight unnaturalness. Context allows for comprehension. May include minor phoneme dropping (1-2 words).
* **Score 2:** Multiple key errors leading to significant comprehension barriers. Frequent mispronunciations, stuttering, or noticeable syllable dropping (≥3 words) that obscure the sentence's core meaning.
* **Score 1:** Systemic errors. Severe and frequent mispronunciations, massive phoneme dropping/adding (≥5 words), making the speech largely incomprehensible or completely misaligned with the text.

### 2. Audio Clarity
**Definition:** Evaluates the physical signal quality, identifying background noise, electronic distortion, reverb, vocal jitter, hoarseness, and other non-textual acoustic artifacts.
* **Score 5:** Studio-quality audio. Clean and clear, free of perceptible background noise, distortion, or electrical hum. Word boundaries are distinct. Minor natural breaths are acceptable.
* **Score 4:** Highly intelligible but with slight, acceptable artifacts (e.g., a stationary noise floor, faint environmental noise, or slightly muffled sound) that do not demand extra listening effort.
* **Score 3:** Noticeable noise, reverb, or slight metallic distortion that occasionally masks phoneme boundaries. Vocals may occasionally sound trembling or hoarse. The listener needs to focus more, but the speech remains understandable.
* **Score 2:** Poor quality. Constant background noise, strong reverb, frequent popping, or severe vocal jitter/hoarseness that frequently masks the main speaker. Listening is strenuous.
* **Score 1:** Severe defects. Extreme distortion, clipping, or overwhelming noise making large portions of the audio impossible to transcribe or understand.

### 3. Intonation Match
**Definition:** Assesses whether the pitch contours match the syntactical structure and context (e.g., declarative, interrogative, imperative, conditional). Evaluates natural pitch fluctuations, ensuring non-terminal intonation for mid-list items and terminal intonation for the final item, strictly penalizing "robotic" or entirely flat generation.
* **Score 5:** Intonation perfectly matches sentence types. Clear cues for questions/transitions. Natural continuous fluctuations without any abrupt turns.
* **Score 4:** Generally good match. Only 1-2 boundary trends might sound slightly stiff, but the overall contour remains mostly natural and doesn't hinder sentence type recognition.
* **Score 3:** Several inappropriate contours or overall monotonicity. Mild mismatch between intonation and text (e.g., a question sounds like a statement). Listener requires context to infer the correct tone.
* **Score 2:** Systemic mismatch. Intonation severely contradicts the sentence structure. Overwhelming "robotic" or "mechanical" prosody.
* **Score 1:** Intonation completely distorts semantics (e.g., extremely exaggerated or sarcastic tone entirely misaligning with the intended meaning).

### 4. Pauses
**Definition:** Evaluates whether pauses align with syntactic/semantic boundaries (clauses, lists, number groupings) and whether their durations are appropriate. Penalizes rushed, "word-by-word" reading styles.
* **Score 5:** Pauses perfectly align with grammatical boundaries. Micro-pauses and clause breaks are natural, providing a clear structural hierarchy.
* **Score 4:** Good overall. Occasional slightly unnatural boundaries that do not hinder comprehension.
* **Score 3:** Noticeable misalignments or inappropriate durations. The listener requires slight cognitive effort to track the sentence structure.
* **Score 2:** Multiple misplacements or a fragmented "word-by-word" reading style. Obvious cognitive load for the listener.
* **Score 1:** Chaotic or missing pauses making the syntactic structure completely unrecognizable.

### 5. Speech Rate
**Definition:** Assesses whether the speaking speed suits the content's context and intent, and whether intra-sentence speed variations are justified (e.g., slightly slower for emphasis, faster for transitions).
* **Score 5:** Speed perfectly matches the content. Justified and natural variations enhance structural clarity without impacting comprehension.
* **Score 4:** Stable and suitable speed. Minor instances of slightly fast or slow segments that do not burden comprehension.
* **Score 3:** Noticeable speed fluctuations. Segments may feel rushed or dragged, requiring the listener to expend effort to keep up, though meaning is retained via context.
* **Score 2:** Unstable speed severely impacting fluency. Frequently too fast or too slow, making parts of the message difficult to catch on the first listen.
* **Score 1:** Severely unbalanced speed control. Extremely fast or slow segments occur frequently, making the majority of the speech incomprehensible.

### 6. Speaker Consistency
**Definition:** Monitors the stability of the speaker's identity (timbre, vocalization method, accent) throughout the utterance. Prevents unexpected identity switching.
* **Score 5:** Highly stable timbre. The listener has absolute certainty it is the same speaker throughout.
* **Score 4:** Good consistency with only subtle variations (e.g., slight brightness/darkness shifts due to effort), easily attributed to natural human variation.
* **Score 3:** Perceptible shifts (e.g., a segment sounds suddenly deeper or changes accent), giving a slight illusion of a different speaker, though the core voiceprint remains similar.
* **Score 2:** Obvious mutations in timbre, accent, or vocalization. Sounds like two different people taking turns reading.
* **Score 1:** Complete failure of consistency. Frequent switching between entirely different voices.

### 7. Style Consistency
**Definition:** Assesses whether the intended speaking style (e.g., formal news, casual chat, customer service, storytelling) is maintained without abrupt register shifts.
* **Score 5:** A unified style is maintained from start to finish, perfectly matching the intended register.
* **Score 4:** Mostly consistent, with minor, acceptable drifts (e.g., a formal reading sounds slightly casual for a moment).
* **Score 3:** Multiple style swings (e.g., jumping between formal broadcasting and casual chatting). Noticeable incongruity.
* **Score 2:** Frequent and abrupt style jumps. Utterance lacks a coherent register thread.
* **Score 1:** No identifiable stable style. A chaotic mix of exaggerated, mechanical, and mismatched tones.

### 8. Emotion Consistency
**Definition:** Evaluates the stability of the intended emotion. If a transition is required by the text, it must be smooth rather than abrupt.
* **Score 5:** Emotional direction is stable, and intensity changes are logically aligned with the context. Transitions are smooth and natural.
* **Score 4:** Mostly consistent. Minor intensity mismatches (e.g., slightly too flat or slightly overacted in one sentence), but main emotional line is clear.
* **Score 3:** Fluctuating intensity or slight deviations from the core emotion. The emotional thread feels somewhat shaky but acceptable.
* **Score 2:** Frequent abrupt emotional mutations (e.g., switching from extreme excitement to coldness without transition). 
* **Score 1:** Severe emotional chaos. Exaggerated or contradictory emotions (e.g., inappropriate sarcasm) that completely disrupt the intended semantic message.

---

## Part II: Advanced Expressiveness Layer (Score 0-2 Bonus)
This layer acts as a bonus dimension to evaluate the anthropomorphic, human-like qualities of high-performance models. A score of 0 is a neutral baseline, not a penalty.

### 9. Stress
**Definition:** The appropriate emphasis of semantic focal points via pitch, loudness, or duration, without over-stressing non-key information (function words).
* **Score 2:** Clear, well-placed stress on key focal words (via pitch, volume, or duration), effectively enhancing the semantic focus and naturalness without feeling forced.
* **Score 1:** Moderate stress. Generally reasonable but lacks ideal intensity or distribution (e.g., slightly too few stresses, or slightly weak). Helpful, but not highly expressive.
* **Score 0:** No positive gain. The reading is completely neutral/flat, or stress is randomly placed, inappropriate, or applied excessively to function words.

### 10. Lengthening
**Definition:** Intentional syllable duration extension at phrase boundaries, emotional words, or semantic focal points to enhance expression or synthesize a smooth closure. *Note: Natural utterance-final lengthening is considered basic prosody and does not count as expressive lengthening.*
* **Score 2:** Clear, contextually appropriate lengthening (e.g., to express hesitation, emphasis, or a smooth syntactic closure) that significantly boosts expressiveness.
* **Score 1:** Mild lengthening. Generally in the right places but doesn't serve as a core expressive highlight.
* **Score 0:** No positive gain. Neutral rhythm, or lengthening occurs inappropriately on function words/random syllables, causing a dragged feeling.

### 11. Paralinguistics
**Definition:** The natural integration of non-lexical sounds (laughter, sighs, breaths, throat-clearing, yawning, coughing, stutters) that match the context without disrupting the semantic flow or rhythm.
* **Score 2:** Prominent, well-placed paralinguistics that perfectly match the context and emotion, massively enhancing the realism of the speech. Removing them would noticeably flatten the delivery.
* **Score 1:** Minor but appropriate paralinguistic cues (e.g., a faint sigh or breath) that add slight flavor to the audio.
* **Score 0:** No positive gain. A clean, neutral reading without paralinguistics, or the inclusion of unnatural, misplaced, or disruptive noises.

### 12. Emotion Expression
**Definition:** The ability to convey a specific, targeted emotion (e.g., joy, anger, sadness) purely through acoustic cues—including pitch, loudness, speed, spectral brightness, intonation contours, and pause distribution—that align with the text.
* **Score 2:** Highly clear and intense emotional cues. The specific emotion is instantly recognizable through multi-dimensional acoustic features, perfectly matching the text. Flattening these cues would significantly reduce the impact.
* **Score 1:** A mild emotional presence. The direction is roughly correct but the acoustic evidence is relatively weak, unstable, or mostly concentrated on just a few phrases.
* **Score 0:** No reliable emotional cues. The speech is essentially a neutral/functional reading. Scores of 0 are given if the listener must rely solely on the semantic meaning of the text to "guess" the emotion, or if the acoustic emotion contradicts the text.
