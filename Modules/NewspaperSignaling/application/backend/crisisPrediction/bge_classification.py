from transformers import pipeline
import logging as log

log.basicConfig(level=log.DEBUG)

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

class BgeClassification:
    def roberta_classifier(self, text, categories_to_classify, threshold=0.85):
        """

        :param threshold: threshold for score of classification
        :param text: text to classify
        :type text: String

        """
        confidence_scores = {}

        bmc_classification = classifier(text, categories_to_classify, multi_label=False)

        for idx, label in enumerate(bmc_classification['labels']):
            confidence_scores[label] = bmc_classification['scores'][idx]

        return confidence_scores


if __name__ == '__main__':
    # dummy sentence test
    sequence_to_classify = "Das Licht ist sehr angenehm und nicht zu kalt. Mich hat überrascht, dass das Licht doch sehr gut gestreut wird. Die alte, stromfressende Glühbirne vermisse ich also nicht!"
    sequence_to_classify2 = "Das Licht ist sehr kalt und dunkel."
    obj = BgeClassification()
    result = obj.roberta_classifier([sequence_to_classify, sequence_to_classify2], ['negativ', 'positiv', 'neutral'], threshold=0.85)

    print(result)
