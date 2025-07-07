class ConsoleStreamer:
    def __init__(self):
        self.speaker_counter = 1
        self.speaker_mapper = {}
        self.interim_speaker = "Speaker 1"  # 暫定句預設 speaker
        self.last_block_speaker = None
        self.line_count = 0

    def display(self, result):
        if result["type"] == "interim":
            self.print_streaming_line(result["text"])
        elif result["type"] == "final":
            self.finalize_line(result)

    def print_streaming_line(self, text):
        # 每次新的 interim 代表一個新的句子段（預設 speaker 1）
        print(f"⏳ [{self.interim_speaker}] {text}")
        self.line_count += 1

    def finalize_line(self, result):
        # 顯示最終修正（顯示後 speaker 可能變化）
        speaker = result["speaker"]
        print(f"✅ [{speaker}] {result['text']} ({result['start']}~{result['end']}s)")
        self.last_block_speaker = speaker

        # ✅ 下一段的 interim speaker 一律預設回 Speaker 1
        self.interim_speaker = "Speaker 1"
