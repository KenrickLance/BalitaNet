<template>
  <Navbar/>
  <div class="main">
    <router-view></router-view>
  </div>
</template>

<script>
  import Navbar from "./components/Navbar";

  export default {
    components: {
      Navbar,
    },

    created() {
      this.$store.commit('SET_IS_MOBILE_OR_TABLET', this.isMobileOrTablet())
      this.$store.commit('SET_IS_MOBILE', this.isMobile())
      this.$store.commit('SET_IS_TABLET', this.isTablet())
    },

    data() {
      return {
        generatedArticle: null,
      }
    },
    methods: {
      isMobileOrTablet() {
        return screen.width <= 1024;
      },
      isMobile() {
        return screen.width <= 640;
      },
      isTablet() {
        return screen.width > 640 && screen.width <= 1024;
      }
    },
  }
</script>
<style>
img {
  max-width: 100%;
  display: block;
}

.main {
  font-family: Lato, Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: left;
  color: #2c3e50;
  font-size: 1rem;
}
</style>
