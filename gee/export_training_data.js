// Export a Sentinel-2 predictor stack (10 m) and Sentinel-5P daily grids (1 km)
// for offline / GPU training with backend/hyperlocal_aq/models_keras.py.
// Outputs go to Google Drive folder "hyperlocal_aq".

var mumbai = ee.Geometry.Rectangle([72.7794, 18.8937, 72.9747, 19.2683]);  // Greater Mumbai bounding box
var START = '2023-11-01', END = '2024-02-28';          // dry season: fewest clouds
var POLLUTANT = {col: 'COPERNICUS/S5P/OFFL/L3_NO2', band: 'tropospheric_NO2_column_number_density', name: 'NO2'};

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(mumbai).filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
  .median().select(['B2', 'B3', 'B4', 'B8']).divide(10000).toFloat();

Export.image.toDrive({image: s2, description: 'S2_B2348_10m_mumbai', folder: 'hyperlocal_aq',
  region: mumbai, scale: 10, crs: 'EPSG:4326', maxPixels: 1e10});

// one band per day: S5P mean, NaN where cloudy / missing
var days = ee.List.sequence(0, ee.Date(END).difference(ee.Date(START), 'day').subtract(1));
var stack = ee.ImageCollection(days.map(function (d) {
  var t0 = ee.Date(START).advance(d, 'day');
  var c = ee.ImageCollection(POLLUTANT.col).select(POLLUTANT.band).filterDate(t0, t0.advance(1, 'day'));
  return ee.Algorithms.If(c.size().gt(0), c.mean().rename(t0.format('YYYYMMdd')),
                          ee.Image.constant(-9999).rename(t0.format('YYYYMMdd')).updateMask(0));
})).toBands();
Export.image.toDrive({image: stack.toFloat(), description: 'S5P_' + POLLUTANT.name + '_daily_1km_mumbai',
  folder: 'hyperlocal_aq', region: mumbai, scale: 1000, crs: 'EPSG:4326', maxPixels: 1e9});
