<template>
  <h5 class="title">Token Scores</h5>
  <Bar
    :chart-options="chartOptions"
    :chart-data="chartData"
    :chart-id="chartId"
    :dataset-id-key="datasetIdKey"
    :plugins="plugins"
    :css-classes="cssClasses"
    :styles="styles"
    :width="width"
    :height="height"
    :responsive="true"
  />
</template>

<script>
import { Bar } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, BarElement, CategoryScale, LinearScale } from 'chart.js'

ChartJS.register(Title, Tooltip, BarElement, CategoryScale, LinearScale)

export default {
  components: { Bar },

  props: {
    values: {
      type: Array,
      required: true,
    },
    indices: {
      type: Array,
      required: true,
    },
    decodedVocab: {
      type: Object,
      required: true,
    }
  },

  data() {
    return {
      chartId: 'bar-chart',
      datasetIdKey: 'label',
      width: 300,
      height: 300,
      cssClasses: '',
      styles: {},
      plugins: [],
      chartOptions: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
          x:{
            suggestedMax: 1,
            ticks:{
              font: {
                size: 12
              }
            }
          },
          y: {
            ticks: {
              font: {
                size: 12
              }
            }
          },
        }
      }
    }
  },

  computed: {
    roundedValues() {
      return this.values.map(value => Math.round(value * 100)/100)
    },
    formattedIndices() {
      return this.indices.map((item, idx) => {
        if (this.values[idx] < .001) {
          return ''
        }
        return this.decodedVocab[String(item)]
      })
    },
    chartData() {
      return {
        labels: this.formattedIndices,
        datasets: [
          {
            backgroundColor: '#f87979',
            data: this.roundedValues
          }
        ]
      }
    }
  },
}
</script>

<style scoped>
.title {
  text-align: center;
}
</style>