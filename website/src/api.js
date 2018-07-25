import axios from 'axios';

export function sentenceCut(item){
    const params={
        model: item.model,
        dataset: item.dataset,
        sentence: item.sentence
    }
    return axios({
          method: 'post',
          url: 'http://127.0.0.1:5000/api/cws',
          data: params
      })
}