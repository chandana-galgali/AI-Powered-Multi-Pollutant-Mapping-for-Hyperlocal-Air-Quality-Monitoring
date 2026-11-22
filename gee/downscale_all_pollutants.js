// Google Earth Engine Code Editor script: quick-look multi-pollutant downscaling.
// Paste into https://code.earthengine.google.com and press Run.
//
// Fixes compared with the hackathon version (GEE Codes.docx):
//  * uses TROPOMI *tropospheric* NO2 (the old script used the total column,
//    whose band name is different, so NO2 failed);
//  * fits the regression on 1 km means and applies it at 10 m, then adds the
//    1 km residual back, so 10 m maps stay consistent with Sentinel-5P;
//  * includes an intercept, and per-pollutant visualisation ranges.

var mumbai = ee.Geometry.Rectangle([72.7794, 18.8937, 72.9747, 19.2683]);  // Greater Mumbai bounding box
var DATE = '2023-12-10';
var S2_BANDS = ['B2', 'B3', 'B4', 'B8'];

var POLLUTANTS = {
  NO2:    {col: 'COPERNICUS/S5P/OFFL/L3_NO2',    band: 'tropospheric_NO2_column_number_density', min: 0,     max: 2.5e-4},
  SO2:    {col: 'COPERNICUS/S5P/OFFL/L3_SO2',    band: 'SO2_column_number_density',               min: 0,     max: 1e-3},
  CO:     {col: 'COPERNICUS/S5P/OFFL/L3_CO',     band: 'CO_column_number_density',                min: 0.025, max: 0.05},
  HCHO:   {col: 'COPERNICUS/S5P/OFFL/L3_HCHO',   band: 'tropospheric_HCHO_column_number_density', min: 0,     max: 3e-4},
  O3:     {col: 'COPERNICUS/S5P/OFFL/L3_O3',     band: 'O3_column_number_density',                min: 0.11,  max: 0.13},
  AER_AI: {col: 'COPERNICUS/S5P/OFFL/L3_AER_AI', band: 'absorbing_aerosol_index',                 min: -1,    max: 2}
};
var PALETTE = ['1f4e5a', '2f7a74', '6fa56a', 'c9c36a', 'e8a84a', 'd9673b', 'b02e3a', '6e1240'];

var d0 = ee.Date(DATE);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(mumbai).filterDate(d0.advance(-45, 'day'), d0.advance(45, 'day'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
  .median().select(S2_BANDS).divide(10000).clip(mumbai);
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var X = s2.addBands(ndvi).addBands(ndwi);                     // 6 predictors at 10 m
var coarseProj = ee.Projection('EPSG:4326').atScale(1000);
var Xc = X.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 65535})
          .reproject(coarseProj);                              // predictors at 1 km

Map.centerObject(mumbai, 11);
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25}, 'Sentinel-2 true colour', true);

Object.keys(POLLUTANTS).forEach(function (key) {
  var p = POLLUTANTS[key];
  var obs = ee.ImageCollection(p.col).select(p.band)
    .filterDate(d0.advance(-3, 'day'), d0.advance(4, 'day')).mean().rename('y');
  // regression on 1 km cells: y ~ 1 + X
  var fit = ee.Image.constant(1).addBands(Xc).addBands(obs.reproject(coarseProj))
    .reduceRegion({reducer: ee.Reducer.linearRegression({numX: 7, numY: 1}),
                   geometry: mumbai, scale: 1000, bestEffort: true});
  var coef = ee.Array(fit.get('coefficients')).project([0]);
  var pred10 = ee.Image.constant(coef.get([0]));
  ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI'].forEach(function (b, i) {
    pred10 = pred10.add(X.select(b).multiply(ee.Image.constant(coef.get([i + 1]))));
  });
  // residual correction: add back the (smoothly interpolated) 1 km residual
  var predC = pred10.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 65535}).reproject(coarseProj);
  var resid = obs.reproject(coarseProj).subtract(predC).resample('bilinear');
  var down = pred10.add(resid).rename('Downscaled_' + key).clip(mumbai);
  var vis = {min: p.min, max: p.max, palette: PALETTE};
  Map.addLayer(obs.clip(mumbai), vis, key + ' Sentinel-5P (1 km)', false);
  Map.addLayer(down, vis, key + ' downscaled (10 m)', key === 'NO2');
});
