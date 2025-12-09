import os
import pandas as pd
from ast import literal_eval

#I created separate csv for both ng and bruk datasets
for name in ["ng","bruk"]:

    #simple csv
    csv_data = pd.DataFrame(columns=['id', 'tokens', 'ner_tags'])

    for file in os.listdir(f'./data/{name}'):
        #check annotation files only - txt has the same name but different extension
        if file.endswith('.ann'):

            #just reading files
            ann_lines = []
            text = ''
            text_lines = []
            with open(os.path.join(f'./data/{name}', file), 'r', encoding='utf-8') as ann_file:
                ann_lines = ann_file.readlines()
            with open(os.path.join(f'./data/{name}', file[:-4] + '.txt'), 'r', encoding='utf-8') as txt_file:
                text_lines = txt_file.readlines()
            
            #quick fix to handle /n in bruk dataset
            for line in text_lines:
                if line == '\n':
                    text += ' SEPARATOR '
                text += line
            
            #parse and filter out only relevant entities
            entities = []
            for line in ann_lines:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                if parts[1] not in ["PERS", "LOC", "ORG"]:
                    continue
                entities.append(parts[1:])
            
            #get words from text
            words = text.split()
            
            #if no relevant entities, tag all as O
            if not entities:
                ner_tags = [6] * len(words)
                csv_row = {
                    'id': len(csv_data),
                    'tokens': words,
                    'ner_tags': ner_tags
                }
                csv_data.loc[len(csv_data)] = csv_row
                continue

            #align entities with words
            ner_tags = []
            file_offset = 0
            current_entity_index = 0
            

            for word in words:
                #quickfix for separator from above
                if word == 'SEPARATOR':
                    file_offset += 1
                    continue

                word_start = file_offset
                word_end = file_offset + len(word)
                # Check if the current word overlaps with the current entity (Also give +1 as there are signs like < or etc.)
                # We can also do check for end but for example in НКВД and НКВД-исти example its better to check only start to map correctly

                if int(entities[current_entity_index][1]) <= word_start + 1:
                    # if word is at the beginning of an entity or previous tag was O, it's a B- tag
                    if (ner_tags[-1] if ner_tags else None) == 'O' or word_start == int(entities[current_entity_index][1]) or word_start + 1 == int(entities[current_entity_index][1]):
                        ner_tags.append('B-' + entities[current_entity_index][0])
                    #else it's an I- tag
                    else:
                        ner_tags.append('I-' + entities[current_entity_index][0])

                    # If the word end surpasses the entity end, move to the next entity
                    if word_end >= int(entities[current_entity_index][2]):
                        current_entity_index += 1
                else:
                    ner_tags.append('O')
                
                #no need to continue if all entities are processed
                if current_entity_index == len(entities):
                    break

                file_offset += len(word) + 1  # +1 for the space
            
            #remove seperator tokens
            words = [word for word in words if word != 'SEPARATOR']

            #extend ner tags because we break after all entities are processed
            ner_tags.extend(['O']*(len(words)-len(ner_tags)))

            #validation checks
            ner_words = [word for words in entities for word in words[3].split()]

            if not len(ner_words) == len([words[i] for i in range(len(ner_tags)) if ner_tags[i] != 'O']):
                print(f"Mismatch between entity words and NER tags {file} : {len(ner_words)} vs {len([words[i] for i in range(len(ner_tags)) if ner_tags[i] != 'O'])}")
                continue

            if not len(entities) == sum(1 for tag in ner_tags if tag.startswith('B-')):
                print(f"Mismatch between entities and NER tags {file}   : {len(entities)} vs {sum(1 for tag in ner_tags if tag.startswith('B-'))}")
                continue
            
            #map ner tags to integers
            ner_tags_map = {
                'B-LOC': 0, 'B-ORG': 1, 'B-PERS': 2, 'I-LOC': 3, 'I-ORG': 4, 'I-PERS': 5, 'O': 6
            }

            #create csv row
            csv_row = {
                'id': len(csv_data),
                'tokens': words,
                'ner_tags': [ner_tags_map[tag] for tag in ner_tags]
            }
            # append to dataframe
            csv_data.loc[len(csv_data)] = csv_row

    #save to csv
    csv_data.to_csv(f'{name}_ner.csv', index=False)
        