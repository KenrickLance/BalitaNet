import { createStore } from 'vuex'

export default createStore({
  state: {
    isMobile: null,
    disableAttention: false,
    disableGenerationSettings: true,
    currentImage: null,
  },
  mutations: {
    SET_IS_MOBILE(state, data) {
      state.isMobile = data
    },
    SET_IS_MOBILE_OR_TABLET(state, data) {
      state.isMobileOrTablet = data
    },
    SET_IS_TABLET(state, data) {
      state.isTablet = data
    },
    SET_CURRENT_IMAGE(state, data) {
      state.currentImage = data
    }
  },
  modules: {
  }
})
