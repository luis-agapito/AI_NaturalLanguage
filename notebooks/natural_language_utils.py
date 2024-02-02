import datasets
import transformers

def getDataset(examples):
    #tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print(f'Found {len(examples)} examples')
    list_of_dicts = []
    for example in examples:
        #story, query, answer = '\n'.join(' '.join(s) for s in example.story), ' '.join(example.query), ' '.join(example.answer)
        aux_str = [' '.join(sentence) for sentence in example.story]
        story_str = '. '.join(aux_str) + '.'
        answer_str = example.answer[0] 
        query_str = ' '.join(example.query) + '?'
    
        #print(story_str)
        #print(answer_str)
        #print(query_str)
    
        start_index = story_str.find(answer_str)
        end_index = start_index + len(answer_str)
        #The office is north of the kitchen. The garden is south of the kitchen.
        #print('answer:', story_str[start_index:end_index])
    
        encoding = tokenizer(story_str, 
                             query_str, 
                             truncation=True, 
                             padding=True, 
                             max_length=tokenizer.model_max_length
                             )

        #Indices of the words (tokens) within the sentence that correspond to the answer.
        start_positions = encoding.char_to_token(start_index)
        end_positions = encoding.char_to_token(end_index - 1)
        
        if start_positions is None:
            start_positions = tokenizer.model_max_length
        if end_positions is None:
            end_positions = tokenizer.model_max_length
    
        dic= {'story': story_str,
              'answer': answer_str,
              'query': query_str,
              'input_ids': encoding['input_ids'],
              'input_ids': encoding['input_ids'],
              'attention_mask': encoding['attention_mask'],
              'start_positions': start_positions,
              'end_positions': end_positions}
        list_of_dicts.append(dic)

    #DatasetDict({
    #    train: Dataset({
    #        features: ['question', 'sentences', 'answer', 
    #                   'str_idx', 'end_idx', 'input_ids', 
    #                   'attention_mask', 'start_positions', 'end_positions'],
    #        num_rows: 1000
    #    })
    #    test: Dataset({
    #        features: ['question', 'sentences', 'answer',
    #                   'str_idx', 'end_idx', 'input_ids', 
    #                   'attention_mask', 'start_positions', 'end_positions'],
    #        num_rows: 1000
    #    })
    #})

    #datasets.Dataset.from_list([{"label": 1, "text": "a"}, {"label": 2, "text": "b"}])    
    return datasets.Dataset.from_list(list_of_dicts)    
