<template>
  <h5 class="attention-title mb-3">Attention Color Settings</h5>
  <div>
    <div class="p-1 d-flex align-items-center justify-content-center">
      <span class="average-title me-2">Average</span>
      <VSwatches v-model="averageColor"
                 :swatches="swatches"
                 swatch-size="25"
                 row-length="7"
                 shapes="squares"
                 show-border
                 popover-x="left"
                 popover-y="top"
                 :trigger-style="{ width: '30px', height: '30px' }"
      >
      </VSwatches>
    </div>
    <div class="layers">
      <div class="mt-2 layers-title">
        <span class="me-2">Layers</span>
        <button type="button"
                class="btn btn-outline-secondary btn-sm shadow-none"
                @click="resetLayers">
          reset
        </button>
      </div>
      <div v-for="i in [0,1,2]"
           :key="i"
           class="row m-2">
        <div v-for="j in [1,2,3,4]"
             :key="j"
             class="col-3 d-flex justify-content-center">
          <VSwatches v-model="layersColors[j-1 + i*4]"
                     :swatches="swatches"
                     swatch-size="25"
                     row-length="7"
                     shapes="squares"
                     show-border
                     popover-x="left"
                     popover-y="top"
                     :trigger-style="{ width: '26px', height: '26px' }"
          >
          </VSwatches>
          <span><b class="numbering m-1">{{ j + i*4 }}</b></span>
        </div>
      </div>
    </div>
    <div class="heads">
      <div class="mt-3 heads-title">
        <span class="me-2">Heads</span>
        <button type="button"
                class="btn btn-outline-secondary btn-sm shadow-none"
                @click="resetHeads">
          reset
        </button>
      </div>
        <div v-for="i in [0,1,2]"
           :key="i"
           class="row m-2">
        <div v-for="j in [1,2,3,4]"
             :key="j"
             class="col-3 d-flex justify-content-center">
          <VSwatches v-model="headsColors[j-1 + i*4]"
                     :swatches="swatches"
                     swatch-size="25"
                     row-length="7"
                     shapes="squares"
                     show-border
                     popover-x="left"
                     popover-y="top"
                     :trigger-style="{ width: '26px', height: '26px' }"
          >
          </VSwatches>
          <span><b class="numbering m-1">{{ j + i*4 }}</b></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import VSwatches from "vue3-swatches"

import {normalizeAttentions, transformAttentions, transformAttentionsLayerOrHead} from "../../util/attentionsUtils";
import {blendColors} from "../../util/colorUtils";

export default {
  components: { VSwatches },

  emits: ['transmitColor'],

  props: {
    attentionsData: {
      type: Object,
      required: true,
    },
    specialTokenIDs: {
      type: Array,
      default: () => []
    }
  },

  data() {
    return {
      averageColor: '#FF0000B0',
      layersColors: Array(12).fill('#FFFFFF00'),
      headsColors: Array(12).fill('#FFFFFF00'),
      swatchColors: ['#FF0000B0', '#FFFF00B0', '#00FF00B0', '#00FFFFB0', '#0000FFB0', '#FF00FFB0'],
      cancelColor: '#FFFFFF00',
    }
  },

  watch: {
    colorMap: {
      handler() {
        this.$emit('transmitColor', this.colorMap)
      },
      immediate: true
    }
  },

  computed: {
    swatches() {
      return this.swatchColors.concat(this.cancelColor)
    },
    transformedAttentionsData() {
      // remove special token and normalize to sum 1 | also rearrange layers and heads
      let newAttentions = {}
      newAttentions['all'] = transformAttentions(this.attentionsData['all'], this.specialTokenIDs)
      newAttentions['layer'] = transformAttentionsLayerOrHead(this.attentionsData['layer'], this.specialTokenIDs)
      newAttentions['head'] = transformAttentionsLayerOrHead(this.attentionsData['head'], this.specialTokenIDs)
      return newAttentions
    },
    colorMap() {
      // return dict(token: color)

      /*
      get dict(color: list(dix(val|idx)))
      get dict(color: single dict(val|idx)
      normalize it to sum 1
       */
      let colorMapping = {}
      this.swatches.forEach((item) => {colorMapping[item] = {'val': [], 'idx': []}})
      //average
      let allValAndInd = this.transformedAttentionsData['all']
      allValAndInd['idx'].forEach((token_idx, idx) => {
        if (colorMapping[this.averageColor]['idx'].includes(token_idx)) {
          let idxInColor = colorMapping[this.averageColor]['idx'].indexOf(token_idx)
          colorMapping[this.averageColor]['val'][idxInColor] += allValAndInd['val'][idx]
        } else {
          colorMapping[this.averageColor]['idx'].push(token_idx)
          colorMapping[this.averageColor]['val'].push(allValAndInd['val'][idx])
        }
      })
      // layer
      this.layersColors.forEach((color, idx) => {
        let tokenIndices = this.transformedAttentionsData['layer'][idx]
        tokenIndices['idx'].forEach((token_idx, idx2) => {
          if (colorMapping[color]['idx'].includes(token_idx)) {
            let idxInColor = colorMapping[color]['idx'].indexOf(token_idx)
            colorMapping[color]['val'][idxInColor] += tokenIndices['val'][idx2]
          } else {
            colorMapping[color]['idx'].push(token_idx)
            colorMapping[color]['val'].push(tokenIndices['val'][idx2])
          }
        })
      })

      // head
      this.headsColors.forEach((color, idx) => {
        let tokenIndices = this.transformedAttentionsData['head'][idx]
        tokenIndices['idx'].forEach((token_idx, idx2) => {
          if (colorMapping[color]['idx'].includes(token_idx)) {
            let idxInColor = colorMapping[color]['idx'].indexOf(token_idx)
            colorMapping[color]['val'][idxInColor] += tokenIndices['val'][idx2]
          } else {
            colorMapping[color]['idx'].push(token_idx)
            colorMapping[color]['val'].push(tokenIndices['val'][idx2])
          }
        })
      })

      delete colorMapping[this.cancelColor]
      Object.values(colorMapping).forEach((item) => normalizeAttentions(item))


      // dict(token: color) hue is mix of colors | strength follow bounded growth formula
      // dict(token: list(dict(color|val|)))
      // dict(token: color)
      let tokenMapping = {}
      Object.keys(colorMapping).forEach((color) => {
        let val = colorMapping[color]['val']
        let tokenIndices = colorMapping[color]['idx']
        tokenIndices.forEach((token_idx, idx) => {
          if (token_idx in tokenMapping) {
            tokenMapping[token_idx].push({'color': color, 'val': val[idx]})
          } else {
            tokenMapping[token_idx] = [{'color': color, 'val': val[idx]}]
          }
        })
      })
      for (let key of Object.keys(tokenMapping)) {
        tokenMapping[key] = blendColors(tokenMapping[key])
      }
      return tokenMapping
    }
  },

  methods: {
    resetHeads() {
      this.headsColors.forEach((val, idx, arr) => arr[idx] = '#FFFFFF00')
    },

    resetLayers() {
      this.layersColors.forEach((val, idx, arr) => arr[idx] = '#FFFFFF00')
    }
  }
}
</script>

<style>
.attention-title {
  text-align: center;
}

.average-title {
  font-size: 1.2em;
}

.layers-title {
  text-align: center;
}

.heads-title {
  text-align: center;
}

.numbering {
  font-size: 12px;
}

.vue-swatches__trigger {
  border: 1px solid #999;
}
</style>