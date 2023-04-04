export function blendColorsOld(colorA, colorB, amount) {
  const [rA, gA, bA] = colorA.match(/\w\w/g).map((c) => parseInt(c, 16));
  const [rB, gB, bB] = colorB.match(/\w\w/g).map((c) => parseInt(c, 16));
  const r = Math.round(rA + (rB - rA) * amount).toString(16).padStart(2, '0');
  const g = Math.round(gA + (gB - gA) * amount).toString(16).padStart(2, '0');
  const b = Math.round(bA + (bB - bA) * amount).toString(16).padStart(2, '0');
  return '#' + r + g + b;
}

export function blendColors(colors) {
  if (typeof colors !== 'object') {
    console.log('blendColors got not object', typeof colors, colors)
    return null
  }
  let totalValue = 0
  let totalR = 0
  let totalG = 0
  let totalB = 0
  colors.forEach((color) => {
    let [r, g, b] = color['color'].match(/\w\w/g).map((c) => parseInt(c, 16)).slice(0,3);
    totalR += (r * color['val'])
    totalG += (g * color['val'])
    totalB += (b * color['val'])
    totalValue += color['val']
  })
  totalR = Math.round(totalR / totalValue).toString(16).padStart(2, '0')
  totalG = Math.round(totalG / totalValue).toString(16).padStart(2, '0')
  totalB = Math.round(totalB / totalValue).toString(16).padStart(2, '0')
  const k = 5
  let strength = Math.round((1 - Math.exp(-1 * k * totalValue)) * 255).toString(16).padStart(2, '0')
  return '#' + totalR + totalG + totalB + strength;
}