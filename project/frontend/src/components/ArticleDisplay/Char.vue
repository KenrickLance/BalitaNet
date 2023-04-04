<template>
  <template v-if="!isSpecialToken">
    <div v-if="isImage"
         class="image-div">
      <img v-if="imageURL !== null"
           :src="imageURL"
           class="image"
           :style="imageStyle">
    </div>
    <span v-else-if="isInput"
          :class="{title: isTitle, body: isBody, category: isCategory}"
          :style="{
            backgroundColor: currentColor
          }">
      {{ formattedChar }}
    </span>
    <span v-else
          :class="{title: isTitle, body: isBody, category: isCategory}"
          :style="{
            backgroundColor: currentColor,
            cursor: 'pointer'
          }"
          @click="isInput ? null : $emit('charClick', tokens)"
          @mouseover="$emit('charHover', tokens)"
          @mouseleave="$emit('charHoverLeave')">
      {{ formattedChar }}
    </span>
  </template>
  <span v-if="char === '<|title|>'">
    <br class="br-category">
  </span>
  <span v-if="char === '<|body|>'">
    <br class="br-title">
  </span>
  <span v-if="char === '\n'">
    <br class="br-body">
  </span>
  <span v-if="lastChar && !(char === '<|EOS|>')">...</span>
</template>

<script>
import {blendColors} from "@/util/colorUtils";

export default {
  props: {
    char: {
      type: String,
      required: true,
    },
    tokens: {
      type: Array,
      required: true,
    },
    remark: {
      type: String,
      required: true,
    },
    inputLength: {
      type: Number,
      required: true,
    },
    colorMap: {
      type: Object,
      default: null,
    },
    hoverToken: {
      type: Number,
      default: null,
    },
    currentToken: {
      type: Number,
      default: null,
    },
    lastChar: {
      type: Boolean
    }
  },

  emits: ['charClick', 'charHover', 'charHoverLeave'],

  data() {
    return {
      hoverColor: '#00000040',
      categories: {
        '0': 'News',
        '1': 'Sports',
        '2': 'Entertainment',
        '3': 'Crime',
        '4': 'Other',
      },
    }
  },

  computed: {
    imageURL() {
      return URL.createObjectURL(this.$store.state.currentImage)
    },

    imageStyle() {
      return this.currentColor
          ? {
            boxShadow: `0px 0px 10px 5px ${this.currentColor}`
          }
          : {}
    },

    formattedChar() {
      if (this.isCategory) {
        return this.categories[this.char]
      } else {
        return this.char
      }
    },

    isSpecialToken() {
      return this.remark === 'special'
    },

    isImage() {
      return this.remark === 'image'
    },

    isTitle() {
      return this.remark === 'title'
    },

    isBody() {
      return this.remark === 'body'
    },

    isCategory() {
      return this.remark === 'category'
    },

    isInput() {
      return this.tokens.every(token => token < this.inputLength)
    },

    color() {
      if (!this.colorMap) {
        return null
      }
      let colors = []
      for (let [key, val] of Object.entries(this.colorMap)) {
        if (this.tokens.includes(Number(key))) {
          colors.push(val)
        }
      }
      if (colors.length === 1) {
        return colors[0]
      } else if (colors.length === 2) {
        return blendColors(colors[0], colors[1], .5)
      } else {
        return colors[0]
      }
    },

    currentColor() {
      if ((this.hoverToken && this.tokens.includes(this.hoverToken)) || (this.currentToken && this.tokens.includes(this.currentToken))) {
        return this.hoverColor
      }
      return this.color
    },
  },
}
</script>

<style scoped>
  span {

  }
  .category {
    font-family: Arvo, serif;
    font-weight: 500;
  }

  .title {
    /*background-color:lightgreen;*/
    font-family: Arvo, serif;
    font-size: 38px;
    line-height: 45px;
    color: #424242;
  }

  .body {
    /*background-color:red;*/
    font-size: 19px;
    line-height: 32px;
    color: #424242;
    font-family: Cambo, serif;
  }

  .image-div {
    margin-top: 1.6em;
  }
  .image {
    max-width: 100%;
    max-height: 25em;
  }

  .br-category {
    display: block; /* makes it have a width */
    content: ""; /* clears default height */
    margin-top: .3em; /* change this to whatever height you want it */
  }

  .br-title {
    display: block; /* makes it have a width */
    content: ""; /* clears default height */
    margin-top: 2em; /* change this to whatever height you want it */
  }

  .br-body {
    display: block; /* makes it have a width */
    content: ""; /* clears default height */
    margin-top: 1em; /* change this to whatever height you want it */
  }
</style>