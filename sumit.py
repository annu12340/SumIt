

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

text_str = '''
India is a country in South Asia. It is the seventh-largest country by area, the second-most populous country, and the most populous democracy in the world. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[e] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia.
Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[21] Their long occupation, initially in varying forms of isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[22] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[23] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest, unfolding as the language of the Rigveda, and recording the dawning of Hinduism in India.[24] The Dravidian languages of India were supplanted in the northern regions.[25] By 400 BCE, stratification and exclusion by caste had emerged within Hinduism,[26] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[27] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires based in the Ganges Basin.[28] Their collective era was suffused with wide-ranging creativity,[29] but also marked by the declining status of women,[30] and the incorporation of untouchability into an organised system of belief.[f][31] In south India, the Middle kingdoms exported Dravidian-languages scripts and religious cultures to the kingdoms of southeast Asia.
In the early medieval era, Christianity, Islam, Judaism, and Zoroastrianism put down roots on India's southern and western coasts.[33] Armed invasions from Central Asia intermittently overran India's plains,[34] eventually establishing the Delhi Sultanate, and drawing northern India into the cosmopolitan networks of medieval Islam.[35] In the 15th century, the Vijayanagara Empire created a long-lasting composite Hindu culture in south India.[36] In the Punjab, Sikhism emerged, rejecting institutionalised religion.[37] The Mughal Empire, in 1526, ushered in two centuries of relative peace,[38] leaving a legacy of luminous architecture.[g][39] Gradually expanding rule of the British East India Company followed, turning India into a colonial economy, but also consolidating its sovereignty.[40] British Crown rule began in 1858. The rights promised to Indians were granted slowly,[41] but technological changes were introduced, and ideas of education, modernity and the public life took root.[42] A pioneering and influential nationalist movement emerged,[43] which was noted for nonviolent resistance and led India to its independence in 1947.
India is a secular federal republic governed in a democratic parliamentary system. It is a pluralistic, multilingual and multi-ethnic society. India's population grew from 361 million in 1951 to 1,211 million in 2011.[44] During the same time, its nominal per capita income increased from US$64 annually to US$1,498, and its literacy rate from 16.6% to 74%. From being a comparatively destitute country in 1951,[45] India has become a fast-growing major economy, a hub for information technology services, with an expanding middle class.[46] It has a space programme which includes several planned or completed extraterrestrial missions. Indian movies, music, and spiritual teachings play an increasing role in global culture.[47] India has substantially reduced its rate of poverty, though at the cost of increasing economic inequality.[48] India is a nuclear weapons state, which ranks high in military expenditure. It has disputes over Kashmir with its neighbours, Pakistan and China, unresolved since the mid-20th century.[49] Among the socio-economic challenges India faces are gender inequality, child malnutrition,[50] and rising levels of air pollution.[51] India's land is megadiverse, with four biodiversity hotspots.[52] Its forest cover comprises 21.4% of its area.[53] India's wildlife, which has traditionally been viewed with tolerance in India's culture,[54] is supported among these forests, and elsewhere, in protected habitats.'''


def _create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):

    sentences = sent_tokenize(text)

    sentence_scores = _score_sentences(sentences, freq_table)

    threshold = _find_average_score(sentence_scores)

    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)

    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)
