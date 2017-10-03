from collections import defaultdict

from sklearn.model_selection import train_test_split

q_0 = '-'
q_f = '\n'

unk_word = '<UNK>'

d = 0.75

data_set_filename = 'berp-POS-training.txt'
training_data_filename = 'training_data.txt'
# test_data_filename = 'assgn2-test-set.txt'
# output_filename = 'sulegaon-nikhil-assgn2-test-output.txt'
test_data_filename = 'test-data.txt'
output_filename = 'out.txt'


class InputParser(object):
    def __init__(self, filename):
        self.filename = filename

    def __get_only_sentences(self):
        file = open(self.filename)
        return [line for line in file if line != '\n']

    def __get_all_sentences(self):
        return [line for line in open(self.filename)]

    def get_tags_in_sequence(self):
        tags_in_order = [q_0]
        for row in self.__get_all_sentences():
            row_array = row.split('\t')
            if len(row_array) == 1:
                tags_in_order.append(q_f)
                tags_in_order.append(q_0)
            else:
                tags_in_order.append(row_array[2].strip())
        return tags_in_order[:-1]  # in order to remove the unnecessary q_0 at the end

    def build_training_data(self):
        train_data = defaultdict(dict)
        vocabulary = set()
        for sentence in self.__get_only_sentences():
            position, word, tag = sentence.split('\t')
            word = word.lower().strip()
            tag = tag.strip()
            vocabulary.add(word)
            train_data[tag][word] = train_data.get(tag, {}).get(word, 0) + 1
        return TrainingData(train_data, list(vocabulary))


class TestDataPreProcessor(object):
    def __init__(self, filename, that_training_data):
        self.filename = filename
        self.training_data = that_training_data

    @staticmethod
    def __reset(sentence, sentences):
        sentence.append(q_f)
        sentences.append(sentence)
        return []

    def convert_to_list_of_tokenized_sentences(self):
        sentences_for_output = []
        sentences_for_test_data = []
        sentence_for_output = []
        sentence_for_test_data = []
        for row in open(self.filename):
            row_array = row.split('\t')
            if len(row_array) == 1:
                sentence_for_output = self.__reset(sentence_for_output, sentences_for_output)
                sentence_for_test_data = self.__reset(sentence_for_test_data, sentences_for_test_data)
            else:
                word = row_array[1].strip()
                sentence_for_output.append(word)
                sentence_for_test_data.append(unk_word if self.training_data.is_unknown_word(word) else word)
        return sentences_for_test_data, sentences_for_output


class TestDataOutput(object):
    def __init__(self, sentences):
        self.sentences = sentences

    @staticmethod
    def write_to_file(filename, str_to_save_to_file):
        file = open(filename, 'w')
        file.write(str_to_save_to_file)
        file.close()

    def save_with_tags(self, pos_tags_per_sentence, filename):
        str_to_save_to_file = ''
        for i in range(len(pos_tags_per_sentence)):
            for j in range(len(pos_tags_per_sentence[i])):
                str_to_save_to_file += str.format('{}\t{}\t{}\n', j + 1, self.sentences[i][j],
                                                  pos_tags_per_sentence[i][j])
            str_to_save_to_file += '\n'
        self.write_to_file(filename, str_to_save_to_file)


class Matrix(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, item):
        return self.matrix[item]

    def items(self):
        return self.matrix.items()


class TransitionMatrix(Matrix):
    def __init__(self, matrix, tag_bigram_model, start, end, tag_count_in_sequence):
        super().__init__(matrix)
        self.tag_bigram_count = tag_bigram_model
        self.start = start
        self.end = end
        self.tag_count_in_sequence = tag_count_in_sequence

    def __number_of_times_tag_occurs(self, tag):
        return self.tag_count_in_sequence.get(tag, 0)

    def __number_of_occurring_bigrams_starting_with(self, tag_a):
        return self.start.get(tag_a, 0)

    def __number_of_occurring_bigrams_ending_with(self, tag_a):
        return self.end.get(tag_a, 0)

    def __number_of_bigram_combinations(self):
        return len(self.tag_bigram_count) ** 2

    def get_smoothed_data(self, tag_a, tag_b):
        # return self.matrix[tag_a][tag_b]
        times_tag_occurs = self.__number_of_times_tag_occurs(tag_a)
        term_a = max((self.tag_bigram_count[tag_a][tag_b] - d), 0) / times_tag_occurs
        lambda_parameter = d / times_tag_occurs * self.__number_of_occurring_bigrams_starting_with(tag_a)
        p_continuation = self.__number_of_occurring_bigrams_ending_with(tag_a) / self.__number_of_bigram_combinations()

        return term_a + (lambda_parameter * p_continuation)


class TrainingData(Matrix):
    def __init__(self, matrix, vocabulary_of_corpus):
        super().__init__(matrix)
        self.vocabulary_of_corpus = vocabulary_of_corpus

    def get_unique_tags(self):
        return list(self.matrix.keys())

    def is_unknown_word(self, word):
        return word not in self.vocabulary_of_corpus


class ViterbiMatrix(Matrix):
    def __init__(self, matrix, input_sequence):
        super().__init__(matrix)
        self.input_sequence = input_sequence

    def __get_tuple_with_max_value_in_last_column(self):
        return max([self.matrix[tag][self.input_sequence[-1]] for tag in self.matrix.keys()], key=lambda tup: tup[0])

    def get_best_tag_sequence(self):
        best_tag_sequence = []
        tag = self.__get_tuple_with_max_value_in_last_column()[1]
        for word in reversed(self.input_sequence[1:]):
            tag = self.matrix[tag][word][1]
            best_tag_sequence.append(tag)
        return list(reversed(best_tag_sequence))


class MatrixFactory(object):
    def __init__(self, data, all_tags, tags_in_training_data):
        self.training_data = data
        self.unique_tags = all_tags
        self.tags_in_sequence = tags_in_training_data

    def __build_dictionary_of_dictionary(self):
        unique_tags_with_start_of_sentence = [q_0] + self.unique_tags
        unique_tags_with_end_of_sentence = self.unique_tags + [q_f]
        return dict(((tag_a, dict((tag_b, 0) for tag_b in unique_tags_with_end_of_sentence)) for tag_a in
                     unique_tags_with_start_of_sentence))

    def __get_tags_bigram_count(self):
        tag_bi_gram_count = self.__build_dictionary_of_dictionary()
        for i in range(len(self.tags_in_sequence) - 1):
            if self.tags_in_sequence[i + 1] != q_0:
                tag_bi_gram_count[self.tags_in_sequence[i]][self.tags_in_sequence[i + 1]] += 1
        return tag_bi_gram_count

    def __get_tag_count(self, tag_a):
        return self.tags_in_sequence.count(tag_a)

    def __get_bigram_start_end_matrix(self):
        tag_bigram_count = self.__get_tags_bigram_count()
        bigram_starts_with_tag_matrix = {}
        bigram_ends_with_tag_matrix = {}
        for tag in training_data.get_unique_tags():
            bigram_starts_with_tag_matrix[tag] = len(
                [tag_b for tag_b, count in tag_bigram_count[tag].items() if count > 0])
            bigram_ends_with_tag_matrix[tag] = len([tag_two_count for tag_one, tag_two_count in tag_bigram_count.items()
                                                    if tag_two_count.get(tag, 0) > 0])
        return bigram_starts_with_tag_matrix, bigram_ends_with_tag_matrix

    def __get_tag_count_in_sequence(self):
        tag_count_in_sequence = dict()
        for tag in [q_0] + self.unique_tags + [q_f]:
            tag_count_in_sequence[tag] = self.tags_in_sequence.count(tag)
        return  tag_count_in_sequence

    def build_transition_matrix(self):
        start_mat, end_mat = self.__get_bigram_start_end_matrix()
        tag_count_in_sequence = self.__get_tag_count_in_sequence()
        matrix = self.__build_dictionary_of_dictionary()
        tags_bigram_count = self.__get_tags_bigram_count()
        for tag_a, tag_b_count in tags_bigram_count.items():
            for tag_b, count in tag_b_count.items():
                matrix[tag_a][tag_b] = count / self.__get_tag_count(tag_a)

        return TransitionMatrix(matrix, tags_bigram_count, start_mat, end_mat, tag_count_in_sequence)


    def build_emission_matrix(self):
        emission_probability_of_unk_word = 1 / len(self.training_data.get_unique_tags())
        matrix = dict(self.training_data.matrix)
        for tag, word_count in self.training_data.items():
            for word in word_count.keys():
                matrix[tag][word] = self.training_data[tag][word] / self.__get_tag_count(tag)
            matrix[tag][unk_word] = emission_probability_of_unk_word
        return Matrix(matrix)


class VectorFactor(MatrixFactory):
    def __init__(self, data, all_tags, tags_in_training_data):
        super().__init__(data, all_tags, tags_in_training_data)

    def build_unigram_tag_probability(self):
        vector = dict((tag, 0) for tag in self.unique_tags)
        number_of_tags = len(self.tags_in_sequence)
        for tag in self.unique_tags:
            vector[tag] = self.tags_in_sequence.count(tag) / number_of_tags
        return Matrix(vector)


class BaselineClassifier(object):
    def __init__(self, that_training_data):
        self.training_data = that_training_data

    def classify(self, sentences):
        tag_per_sentence = []
        for sentence in sentences:
            tags = []
            for word in sentence:
                tag_word_count = [(tag, word_count[word]) for tag, word_count in self.training_data.matrix.items() if
                                  word_count.get(word, 0) != 0]
                if len(tag_word_count) > 1:
                    tags.append(max(tag_word_count, key=lambda tup: tup[1])[0])
                else:
                    tags.append('')
            tag_per_sentence.append(tags)
        return tag_per_sentence


class Viterbi(object):
    def __init__(self, transition_probability, emission_probability, all_tags, data):
        self.transition_matrix = transition_probability
        self.emission_matrix = emission_probability
        self.all_tags = all_tags
        self.training_data = data

    def __get_max_value_tag_tuple(self, all_viterbi_values):
        max_viterbi_value = max(all_viterbi_values)
        tag = self.all_tags[all_viterbi_values.index(max_viterbi_value)]
        return max_viterbi_value, tag

    def run(self, sentences):
        viterbi_matrices = []
        for tokenized_sentence in sentences:
            viterbi_matrix = dict((tag, dict(tuple())) for tag in self.all_tags)
            first_word = tokenized_sentence[0]
            for tag in self.all_tags:
                viterbi_matrix[tag][first_word] = (self.emission_matrix[tag].get(first_word, 0) *
                                                   self.transition_matrix.get_smoothed_data(q_0, tag), q_0)

            i = 1
            for word in tokenized_sentence[1:-1]:
                for tag in self.all_tags:
                    all_viterbi_values = [viterbi_matrix[previous_tag][tokenized_sentence[i - 1]][0] *
                                          self.emission_matrix[tag].get(word, 0) *
                                          self.transition_matrix.get_smoothed_data(previous_tag, tag) for previous_tag
                                          in self.all_tags]
                    viterbi_matrix[tag][word] = self.__get_max_value_tag_tuple(all_viterbi_values)
                i += 1

            last_word = tokenized_sentence[-1]
            for tag in self.all_tags:
                all_viterbi_values = [viterbi_matrix[previous_tag][tokenized_sentence[-2]][0] *
                                      self.transition_matrix.get_smoothed_data(previous_tag, q_f) for previous_tag
                                      in self.all_tags]
                viterbi_matrix[tag][last_word] = self.__get_max_value_tag_tuple(all_viterbi_values)

            viterbi_matrices.append(ViterbiMatrix(viterbi_matrix, tokenized_sentence))
        return viterbi_matrices


def split_training_data_to_test_data():
    data_set = []
    sentence = []
    for line in open(data_set_filename):
        if line == '\n':
            data_set.append(sentence)
            sentence = []
        else:
            sentence.append(line)

    training_set, test_set = train_test_split(data_set, test_size=0.1, random_state=132)

    TestDataOutput.write_to_file(training_data_filename, str.join('\n', [str.join('', sent) for sent in training_set]) + '\n')
    TestDataOutput.write_to_file(test_data_filename, str.join('\n', [str.join('', sent) for sent in test_set]) + '\n')


def perform_baseline_classification():
    global processor, test_data_sentences, output_sentences
    processor = TestDataPreProcessor(test_data_filename, training_data)
    test_data_sentences, output_sentences = processor.convert_to_list_of_tokenized_sentences()
    classifier = BaselineClassifier(training_data)
    pos_tags_per_sentence = classifier.classify(test_data_sentences)
    TestDataOutput(output_sentences).save_with_tags(pos_tags_per_sentence, output_filename)


if __name__ == '__main__':
    split_training_data_to_test_data()

    parser = InputParser(training_data_filename)
    training_data = parser.build_training_data()

    # perform_baseline_classification()

    unique_tags = training_data.get_unique_tags()
    tags_in_sequence = parser.get_tags_in_sequence()

    processor = TestDataPreProcessor(test_data_filename, training_data)
    test_data_sentences, output_sentences = processor.convert_to_list_of_tokenized_sentences()

    factory = MatrixFactory(training_data, unique_tags, tags_in_sequence)
    transition_matrix = factory.build_transition_matrix()
    emission_matrix = factory.build_emission_matrix()

    viterbi = Viterbi(transition_matrix, emission_matrix, unique_tags, training_data)
    matrices = viterbi.run(test_data_sentences)

    pos_tags_of_sentences = [matrix.get_best_tag_sequence() for matrix in matrices]

    TestDataOutput(output_sentences).save_with_tags(pos_tags_of_sentences, output_filename)
