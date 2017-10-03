from collections import defaultdict

q_0 = '-'
q_f = '\n'

training_data_filename = 'berp-POS-training.txt'
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
        for sentence in self.__get_only_sentences():
            position, word, tag = sentence.split('\t')
            word = word.lower().strip()
            tag = tag.strip()
            train_data[tag][word] = train_data.get(tag, {}).get(word, 0) + 1
        return TrainingData(train_data)


class TestDataPreProcessor(object):
    def __init__(self, filename):
        self.filename = filename

    def convert_to_list_of_sentences(self):
        sentences = []
        sentence = []
        for row in open(self.filename):
            row_array = row.split('\t')
            if len(row_array) == 1:
                sentence.append(q_f)
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(row_array[1].strip())
        return sentences


class TestDataOutput(object):
    def __init__(self, sentences):
        self.sentences = sentences

    @staticmethod
    def _write_to_file(filename, str_to_save_to_file):
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
        self._write_to_file(filename, str_to_save_to_file)


class Matrix(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, item):
        return self.matrix[item]

    def items(self):
        return self.matrix.items()


class TransitionMatrix(Matrix):
    def __init__(self, matrix):
        super().__init__(matrix)

    def get_smoothed_data(self, tag_a, tag_b):
        return self.matrix[tag_a][tag_b]



class TrainingData(Matrix):
    def __init__(self, matrix):
        super().__init__(matrix)

    def get_unique_tags(self):
        return list(self.matrix.keys())


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

    def build_transition_matrix(self):
        matrix = self.__build_dictionary_of_dictionary()
        for tag_a, tag_b_count in self.__get_tags_bigram_count().items():
            for tag_b, count in tag_b_count.items():
                matrix[tag_a][tag_b] = count / self.__get_tag_count(tag_a)
        return TransitionMatrix(matrix)

    def build_emission_matrix(self):
        matrix = dict(self.training_data.matrix)
        for tag, word_count in self.training_data.items():
            for word in word_count.keys():
                matrix[tag][word] = self.training_data[tag][word] / self.__get_tag_count(tag)
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


class Classifier(object):
    def __init__(self, that_training_data):
        self.training_data = that_training_data

    def classify(self, words):
        y = []
        for index, word in enumerate(words):
            prediction = ''
            tags_with_counts = self.training_data.get(word.lower(), {})
            if len(tags_with_counts) > 0:
                prediction = max(tags_with_counts, key=tags_with_counts.get)
            y.append('{}\t{}\t{}\n'.format(index + 1, word, prediction))
        return y


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
            viterbi_matrix = dict(
                (tag, (dict((word, tuple()) for word in tokenized_sentence))) for tag in self.all_tags)
            first_word = tokenized_sentence[0]
            for tag in self.all_tags:
                viterbi_matrix[tag][first_word] = (self.emission_matrix[tag].get(first_word, 0) *
                                                   self.transition_matrix.get_smoothed_data(q_0, tag), q_0)

            i = 1
            for word in tokenized_sentence[1:-1]:
                for tag in self.all_tags:
                    all_viterbi_values = [viterbi_matrix[previous_tag][tokenized_sentence[i - 1]][0] *
                                          self.emission_matrix[tag].get(word, 0) *
                                          self.transition_matrix.get_smoothed_data(previous_tag, tag) for previous_tag in self.all_tags]
                    viterbi_matrix[tag][word] = self.__get_max_value_tag_tuple(all_viterbi_values)
                i += 1

            last_word = tokenized_sentence[-1]
            for tag in self.all_tags:
                all_viterbi_values = [viterbi_matrix[previous_tag][tokenized_sentence[-2]][0] *
                                      self.transition_matrix.get_smoothed_data(previous_tag, q_f) for previous_tag in self.all_tags]
                viterbi_matrix[tag][last_word] = self.__get_max_value_tag_tuple(all_viterbi_values)

            viterbi_matrices.append(ViterbiMatrix(viterbi_matrix, tokenized_sentence))
        return viterbi_matrices


if __name__ == '__main__':
    parser = InputParser(training_data_filename)
    training_data = parser.build_training_data()
    unique_tags = training_data.get_unique_tags()
    tags_in_sequence = parser.get_tags_in_sequence()

    processor = TestDataPreProcessor(test_data_filename)
    test_data_sentences = processor.convert_to_list_of_sentences()

    factory = MatrixFactory(training_data, unique_tags, tags_in_sequence)
    transition_matrix = factory.build_transition_matrix()
    emission_matrix = factory.build_emission_matrix()

    viterbi = Viterbi(transition_matrix, emission_matrix, unique_tags, training_data)
    matrices = viterbi.run(test_data_sentences)

    pos_tags_of_sentences = [matrix.get_best_tag_sequence() for matrix in matrices]

    TestDataOutput(test_data_sentences).save_with_tags(pos_tags_of_sentences, output_filename)

    # print("\t" + "\t".join(x for x in unique_tags))
    # print("".join(["-"] * 90))
    # for ii in [q_0] + unique_tags:
    #     print("%s:\t" % ii + "\t".join(str(transition_matrix[ii].get(x, 0))
    #                                    for x in unique_tags))

    # l = classify(['I', 'am', 'Nikhil'])
    # for pred in l:
    #     print(pred)
