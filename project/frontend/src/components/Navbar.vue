<template>
  <nav class="navbar sticky-top navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <router-link class="navbar-brand me-3" to="/">BalitaNet</router-link>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav mr-auto me-auto mb-2 mb-lg-0">
          <li class="nav-item" >
            <router-link 
            :class="{ active: currentRouteIsHome }"
            class="nav-link" to="/">Generate Articles </router-link>
          </li>
          <li class="nav-item" :class="{ active: currentRouteIsAbout }">
            <router-link
            :class="{ active: currentRouteIsAbout }"
            class="nav-link" to="/about">About</router-link>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  computed: {
    currentRouteName() {
      return this.$route.name;
    },

    currentRouteIsHome() {
      return this.currentRouteName === 'Home';
    },

    currentRouteIsAbout() {
      return this.currentRouteName === 'About';
    }

  }
}
</script>