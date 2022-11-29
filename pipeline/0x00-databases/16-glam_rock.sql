-- Lists all bands with GLam rock as their main style
SELECT band_name, (IFNULL(split, 2020) - formed) AS lifespan FROM metal_bands
WHERE style LIKE '%glam rock%'
ORDER BY lifespan DESC;