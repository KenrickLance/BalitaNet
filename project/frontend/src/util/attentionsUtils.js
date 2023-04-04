export function transformAttentions(attentions, specialTokenIDs) {
  let newAttentions = {'val': [], 'idx': []}

  // remove special tokens
  attentions['idx'].forEach((token_idx, idx) => {
    if (!specialTokenIDs.includes(token_idx)) {
      newAttentions['val'].push(attentions['val'][idx])
      newAttentions['idx'].push(token_idx)
    }
  })

  // sum to 1
  normalizeAttentions(newAttentions)

  return newAttentions
}

export function transformAttentionsLayerOrHead(attentions, specialTokenIDs) {
  let newAttentions = []
  attentions['idx'].forEach((item, idx) => {
    newAttentions.push(transformAttentions({'val': attentions['val'][idx], 'idx': item}, specialTokenIDs))
  })
  return newAttentions
}

export function normalizeAttentions(attentions) {
  if (attentions['val'].length === 0) {
    return
  }
  let sum = attentions['val'].reduce((a, b) => a + b)
  attentions['val'].forEach((item, idx, arr) => {
    arr[idx] = Math.round(item / sum * 1000) / 1000
  })
}