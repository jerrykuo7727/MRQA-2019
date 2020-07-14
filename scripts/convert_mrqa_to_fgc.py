import sys
import gzip
import json


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print('Usage: python3 convert_mrqa_to_fgc.py <mrqa-format-input-fpath> <fgc-format-output-fpath>')
        exit(1)
        
    input_fpath = sys.argv[1]
    output_fpath = sys.argv[2]
    
    
    # Read MRQA-format data
    with gzip.open(input_fpath) as f:
        jsonl_data = f.readlines()
        
    data_info = json.loads(jsonl_data[0])
    dataset = data_info['header']['dataset']
    line_count = len(jsonl_data) - 1

    
    # Convert MRQA-format to FGC-format
    new_data = []
    for di, jsonl_line in enumerate(jsonl_data[1:], start=1):
        
        # PQA (Outer loop)
        new_PQA = {}
        PQA = json.loads(jsonl_line)
        DID = '%d' % di
        DTEXT = PQA['context']
        new_PQA['DID'] = DID
        new_PQA['DTEXT'] =  DTEXT
        new_PQA['QUESTIONS'] = []

        # QA (Middle loop)
        for qi, QA in enumerate(PQA['qas'], start=1):
            new_QA = {'AMODE': 'Single-Span-Extraction', 'ATYPE': ''}
            QID = '%s-%d' % (DID, qi)
            QTEXT = QA['question']
            new_QA['QID'] = QID
            new_QA['QTEXT'] = QTEXT

            # Inner A (Inner loop)
            answer_map = {}
            new_ANSWER, new_ASPAN = [], []
            for A in QA['detected_answers']:
                ATEXT = A['text']
                start = A['char_spans'][0][0]
                end = A['char_spans'][0][1]

                # ANSWER
                if ATEXT not in answer_map:
                    answer_map[ATEXT] = len(answer_map)
                    new_ANSWER.append({'ATEXT': ATEXT, 'ATOKEN': [{'text': ATEXT, 'start': start}]})
                else:
                    ai = answer_map[ATEXT]
                    atoken_info = {'text': ATEXT, 'start': start}
                    if atoken_info not in new_ANSWER[ai]['ATOKEN']:
                        new_ANSWER[ai]['ATOKEN'].append(atoken_info)

                # ASPAN
                aspan_info = {'text': ATEXT, 'start': start, 'end': end}
                if aspan_info not in new_ASPAN:
                    new_ASPAN.append(aspan_info)

            new_QA['ANSWER'] = new_ANSWER
            new_QA['ASPAN'] = new_ASPAN
            new_PQA['QUESTIONS'].append(new_QA)
        new_data.append(new_PQA)
        print('%s: %d/%d (%.2f%%)\r' % (dataset, di, line_count, 100*di/line_count), end='')
    print()
    
    # Save FGC-format data as JSON
    with open(output_fpath, 'w') as f:
        json.dump(new_data, f)
