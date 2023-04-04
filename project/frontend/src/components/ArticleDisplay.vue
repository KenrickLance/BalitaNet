<template>

  <div class="container-lg" style="min-height: 100vh">
    <div class="row">
      <div :class="char2tokenClass">
        <template v-if="!$store.state.isMobileOrTablet && !$store.state.disableAttention">
          <Char v-for="(item, index) in char2token"
                :key="index"
                v-bind="item"
                :inputLength="inputLength"
                :colorMap="colorMap"
                :hoverToken="item['tokens'].includes(hoverToken) ? hoverToken : null"
                :currentToken="item['tokens'].includes(currentToken) ? currentToken : null"
                :lastChar="index === char2token.length-1"
                @charClick="charClick"
                @charHover="charHover"
                @charHoverLeave="charHoverLeave"
          />
        </template>
        <template v-else>
          <Char v-for="(item, index) in char2token"
                :key="index"
                v-bind="item"
                :inputLength="char2token.length"
                :lastChar="index === char2token.length-1"
          />
        </template>
      </div>
      <div v-if="!$store.state.isMobileOrTablet && !$store.state.disableAttention" class="col-4">
        <div v-if="currentToken" class="position-relative">
          <a @click.prevent="clearCurrentToken"
             href="#"
             class="close position-absolute"></a>
          <div class="mt-3">
            <Scores v-bind="scoresData"
                    :decodedVocab="decodedVocab"/>
          </div>
          <div class="mt-4 pt-3">
            <Attentions  :attentionsData="attentionsData"
                         :specialTokenIDs="specialTokenIDs"
                         @transmitColor="(x) => colorMap = x"
            />
          </div>
        </div>
        <div v-else class="noToken mt-3 ms-5 me-5 mb-5">
          Select text to view token and attention scores
        </div>
      </div>
    </div>
  </div>
  &nbsp;
</template>

<script>
  import Char from "@/components/ArticleDisplay/Char"
  import Scores from "@/components/ArticleDisplay/Scores"
  import Attentions from "@/components/ArticleDisplay/Attentions"

  import decoded_vocab from "@/vocabs/decoded_vocab.json"

  export default {
    components: {
      Char,
      Scores,
      Attentions
    },

    props: {
      char2token: {
        type: Object,
        required: true
      },
      scores: {
        type: Object,
      },
      attentions: {
        type: Object,
      },
      inputLength: {
        type: Number,
        required: true
      },
    },

    data() {
      return {
        decodedVocab: decoded_vocab,

        colorMap: null,

        currentTokens: null,
        currentToken: null,
        hoverTokens: null,
        hoverToken: null,
      }
    },

    watch: {
      char2token() {
        this.clearCurrentToken()
      }
    },

    computed: {
      char2tokenClass() {
        if (this.$store.state.isMobileOrTablet || this.$store.state.disableAttention) {
          return {
            'col-12': true,
          }
        } else {
          return {
            'col-8': true,
            'border-end': true,
            'border-2': true,
            'border-secondary': true,
          }
        }
      },
      scoresData() {
        return this.scores[this.currentToken - this.inputLength]
      },
      attentionsData() {
        return this.attentions[this.currentToken - this.inputLength]
      },
      specialTokenIDs() {
        let tokens = []
        for (let x of this.char2token) {
          if (x['remark'] === 'special') {
            for(let token of x['tokens']) {
              tokens.push(token)
            }
          }
        }
        return tokens
      },
    },

    methods: {
      charClick(tokens) {
        this.currentTokens = tokens
        if (tokens.length === 1) {
          this.currentToken = tokens[0]
        } else {
          this.currentToken = tokens[0]
        }
      },

      charHover(tokens) {
        this.hoverTokens = tokens
        if (tokens.length === 1) {
          this.hoverToken = tokens[0]
        } else {
          this.hoverToken = tokens[0]
        }
      },

      charHoverLeave() {
        this.hoverTokens = null
        this.hoverToken = null
      },

      clearCurrentToken() {
        this.colorMap = null
        this.currentTokens = null
        this.currentToken = null
        this.hoverTokens = null
        this.hoverToken = null
      },
    },
  }

</script>

<style scoped>
.noToken {
  color: #000;
  color: rgba(0, 0, 0, .5);
  text-align: center;
  font-size: 1.5em;
  font-family: Arial, sans-serif;
}

.close {
  position: absolute;
  right: -18px;
  top: -15px;
  width: 21px;
  height: 21px;
  opacity: 0.3;
}
.close:hover {
  opacity: 1;
}
.close:before, .close:after {
  position: absolute;
  left: 2px;
  content: ' ';
  height: 22px;
  width: 2px;
  background-color: #333;
}
.close:before {
  transform: rotate(45deg);
}
.close:after {
  transform: rotate(-45deg);
}
</style>