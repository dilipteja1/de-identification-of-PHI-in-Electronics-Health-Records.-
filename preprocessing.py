import nltk


def extract_tags(file):
    '''
        function to extract tags from the xml files. tags for the text is
        present in the TAGS attribute
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    PHI_category = ['NAME', 'PROFESSION', 'LOCATION', 'AGE', 'DATE', 'CONTACT',
                    'ID']  # Here are the seven PHI category defined by i2b2      #PHI_category=[category]
    tag_list = []  # An empty list to hold all dictionary items
    for category in PHI_category:
        for tag in root.iter():
            if tag.tag == category:  # skip if a specific tag is not found
                tag.attrib['Category'] = category  # add a column on category
                tag.attrib['File'] = file[len(file) - 10:len(file) - 4]  # add a column to indicate file name
                tag_list.append(tag.attrib)
    temp_df = pd.DataFrame(tag_list)

    print(temp_df)
    return temp_df


def extract_notes(file):
    '''
        extract the notes present in the text. it gets the text from
        the text tag and removes all the spaces and new lines and keep the
        sentences
    :param file:
    :return:
         list of all notes of the medical record
    '''
    # '''This function breakdown inidividaul EHR text note into sentences using XML tags, divided by new line and period'''
    tree = ET.ElementTree(file=file)
    root = tree.getroot()
    all_notes = []

    text = root.find('TEXT').text
    sentences = [sent.split('\n') for sent in sent_tokenize(text) if sent != '\n']

    for text in sentences:  # this part ignore empty lines
        for sub_item in text:
            if sub_item.replace(' ', '') != '':
                all_notes.append(sub_item)

    print(all_notes)

    return all_notes


def sentence_encoding(file):
    """
        annotates sentence with the corresponding NER word and its type
        and category
    :param file:
        the health record file
    :return:
        dataframe with annotations for the sentences
    """
    sentence_list = extract_notes(file)  # generate a list of sentences from tex

    df = extract_tags(file)
    text_list = df['text'].tolist()  # generate a list of tag "TEXT"
    type_list = df['TYPE'].tolist()  # generate a list of tag "type"
    category_list = df['Category'].tolist()  # generate a list of tag "category"

    processed_sentence = []
    processed_text = []
    processed_type = []
    processed_category = []

    def findWholeWord(
            w):  # this function finds a word within a string broken down by regular expression (case sensitive)
        return re.compile(r'\b({0})\b'.format(w)).search

    for sentence in sentence_list:
        for text in text_list:
            if findWholeWord(text)(sentence) != None:
                processed_sentence.append(sentence)
                processed_text.append(text)
                processed_type.append(type_list[text_list.index(text)])
                processed_category.append(category_list[text_list.index(text)])

    temp_df = pd.DataFrame({'Sentence': processed_sentence, 'Word': processed_text, 'Type': processed_type,
                            'Category': processed_category})
    df = temp_df.drop_duplicates()

    return df


def token_annotator(file):
    temp_df = sentence_encoding(file)  # take the data frame and turn them into individual lists

    type_list = temp_df['Type'].tolist()
    temp_sentence_list = temp_df['Sentence'].tolist()
    word_list = temp_df['Word'].tolist()
    temp_unique_sentence_list = set(temp_sentence_list)
    sentence_list = list(temp_unique_sentence_list)  # take out duplicate sentences

    tokenized_word = []  # separate individual text into words (e.g, Mia E. Tapia to "Mia","E.","Tapia")
    tokenizer = get_tokenizer()
    for phrase in word_list:
        tokenized_word.append(tokenizer.tokenize(phrase))

    tokenized_sentence = []
    encoded_token = []

    for i in range(len(sentence_list)):  # tokenize the sentence and encode individual word
        token_list = tokenizer.tokenize(sentence_list[i])
        tokenized_sentence.append(token_list)
        temp_list = ['O' for length in range(len(token_list))]
        for j in range(len(tokenized_word)):
            if all(elem in token_list for elem in tokenized_word[j]) == True:
                # print(token_list, tokenized_word[j])
                for word in tokenized_word[j]:
                    temp_list[token_list.index(word)] = (type_list[j])
                    # print(temp_list)
        encoded_token.append(temp_list)

    return tokenized_sentence, encoded_token


def type_token_generator(file, tokenizer):
    # this function convert all the text of a record into individual BERT tokenized list and generate type encoding list
    all_sentences = extract_notes(file)
    tokenized_sentences = []
    for sentence in all_sentences:
        tokenized_sentences.append(tokenizer.tokenize(sentence))

    type_token = []

    sentence_list, encoded_token = token_annotator(file)

    for sentence in tokenized_sentences:
        if sentence in sentence_list:
            type_token.append(encoded_token[sentence_list.index(sentence)])
        else:
            type_token.append(['O' for i in range(len(sentence))])

    label_list = []
    label_dict = {"O": 0, "DATE": 1, "DOCTOR": 2, "HOSPITAL": 3, 'PATIENT': 4, 'AGE': 5, 'MEDICALRECORD': 6, 'CITY': 6,
                  'STATE': 6, 'PHONE': 6, 'USERNAME': 6, 'IDNUM': 6, 'PROFESSION': 6, 'STREET': 6, 'ZIP': 6,
                  'ORGANIZATION': 6, 'COUNTRY': 6, 'FAX': 6, 'DEVICE': 6, 'EMAIL': 6, 'LOCATION-OTHER': 6, 'URL': 6,
                  'HEALTHPLAN': 6, 'BIOID': 6}  # ,'IPADDRESS':24,'ACCOUNT NUMBER':25}
    for type_list in type_token:  # we convert the label to numerical for Bert training. We can add types here later.
        label_list.append([label_dict.get(item, item) for item in type_list])
        # label_list.append([0 if typetoken =='O' else 1 for typetoken in type_list])

    tokenized_sentences1 = []
    type_token1 = []
    label_list1 = []

    # Remove sentence with only 0
    for i in range(len(label_list)):
        if not all(token == 0 for token in label_list[i]):
            label_list1.append(label_list[i])
            tokenized_sentences1.append(tokenized_sentences[i])
            type_token1.append(type_token[i])

    return tokenized_sentences1, type_token1, label_list1


def bert_array(file, max_seq_length, tokenizer):
    '''This function generates the 5 lists of array that is required to feed into the model'''

    token_sentence, type_token, label_list = type_token_generator(file)

    token_list = []
    input_IDs = []
    input_mask = []  # 1 for non padding and 0 for padding
    segment_ID = []
    label = []

    for untrimmed_sentence in token_sentence:
        sentence = untrimmed_sentence[0:(max_seq_length) - 2]  # trim the list to allow space for CLS and SEP
        sentence.insert(0, '[CLS]')
        sentence.insert(len(sentence), '[SEP]')
        length_before_padding = len(sentence)
        temp_inputID = [1 for i in range(length_before_padding)]  # insert 1 for [CLS] and [SEP] for mask
        sentence.extend(['[PAD]' for i in range(max_seq_length - len(sentence))])
        temp_inputID.extend([0 for i in range(max_seq_length - len(temp_inputID))])
        token_list.append(sentence)
        input_mask.append(temp_inputID)
        segment_ID.append([0 for i in range(max_seq_length)])

    for token in token_list:
        input_ids = tokenizer.convert_tokens_to_ids(token)
        input_IDs.append(input_ids)

    for untrimmed_item in label_list:
        item = untrimmed_item[0:(max_seq_length - 2)]  # trim the list to allow space for CLS and SEP
        item.insert(0, 7)  # 7 for CLS class label 24 for CLS (Arnobio - you need to change 24 to 0 for binary)
        item.insert(len(item), 8)  # 8 for CLS class label 25 for SEP (Arnobio - you need to change 25 to 0 for binary)
        item.extend([9 for i in range(max_seq_length - len(
            item))])  # 9 for PAD class label 26 represents paddinging (ARnobio you need to change 26 to 0 for binary)
        label.append(item)

    return token_list, input_IDs, input_mask, segment_ID, label


def generate_train_data(training_files, max_seq_length=20):  # Max number of file number is 789
    """
        This function runs through a loop to append the tokens, input ids, input masks, segement id and labels to 5
        individual np arrays
    """
    temp_list0, temp_list1, temp_list2, temp_list3, temp_list4 = [], [], [], [], []
    for file in training_files:
        temp_data = bert_array(file, max_seq_length, tokenizer=nltk.NLTKWordTokenizer)

        for j in range(len(temp_data[0])):
            temp_list0.append(temp_data[0][j])
            temp_list1.append(temp_data[1][j])
            temp_list2.append(temp_data[2][j])
            temp_list3.append(temp_data[3][j])
            temp_list4.append(temp_data[4][j])

    #   np_token_list=np.array(temp_list0)
    #   np_input_ids=np.array(temp_list1)
    #   np_input_masks=np.array(temp_list2)
    #   np_segment_ids=np.array(temp_list3)
    #   np_labels=np.array(temp_list4)

    # return np_token_list, np_input_ids, np_input_masks, np_segment_ids, np_labels
    return temp_list0, temp_list1, temp_list2, temp_list3, temp_list4

# change number of file here (MAX:789)
