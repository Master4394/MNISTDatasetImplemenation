from googletrans import Translator

def translate_text(text, src_lang, dest_lang):
    """Translates text between English and Swahili using Google Translate API."""
    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

# Test translations
english_text = "Hello, how are you?"
swahili_text = "Habari, uko aje?"

# Translate English to Swahili
print("English to Swahili:", translate_text(english_text, "en", "sw"))

# Translate Swahili to English
print("Swahili to English:", translate_text(swahili_text, "sw", "en"))
