import numpy as np
import pdb

def beam_search(input_dist, beamsize=10, maxlenratio=1.0):
    '''
        params:
            input_dist: (T, V), T is timestamp, V is vocabulary size
            beamsize: beam size
            maxlenratio: max length ratio, actual length = int(maxlenratio*T)
    '''
    T, V = input_dist.shape
    beam = [([], 0)]
    for t in range(max(1, int(maxlenratio*T))):
        next_beam = []

        for seq, score in beam:
            if len(seq) > 0:
                last_token = seq[-1]
            else:
                last_token = None

            for v in range(V):
                if last_token is None:
                    candidate_score = input_dist[t][v]
                else:
                    candidate_score = score + input_dist[t][v]
            
                candidate_seq = seq + [v]
                next_beam.append((candidate_seq, candidate_score))

        next_beam.sort(key=lambda x: x[1], reverse=True)

        beam = next_beam[:beamsize]

    return beam
        
def ctc_decode_post_process(ctc_result, charlist="_abcdefghijklmnopqrstuvwxyz", blank_label=0):
    '''
        params:
            ctc_result (list): List of integers representing CTC-decoded sequence.
            blank_label (int): Label index for the "blank" character.
        return:
            text: The final text sequence with "blank" labels removed.
    '''
    result = []
    prev_label = None

    for label in ctc_result:
        if label != blank_label and label != prev_label:
            result.append(label)
        prev_label = label

    text = "".join([charlist[id] for id in result])
    return text

if __name__ == '__main__':
    dist = np.log(np.array([[0.2, 0.3, 0.5],
                            [0.4, 0.4, 0.2],
                            [0.1, 0.3, 0.6], 
                            [0.5, 0.2, 0.3]]))
    
    final_beam = beam_search(dist, beamsize=2, maxlenratio=1.0)
    for seq, score in final_beam:
         print("W/o postprocess sequence:", seq, "Probability: ", np.exp(score))  
         #b_b_  ->  bb
         #bab_  ->  bab
         print("final text: ", ctc_decode_post_process(seq, charlist="_ab"))
