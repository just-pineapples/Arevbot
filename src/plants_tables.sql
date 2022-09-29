create TABLE assets(
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    ppa_rate INTEGER NOT NULL,
    asset_type TEXT NOT NULL,
    ac_capacity INTEGER NOT NULL,
    dc_capacity INTEGER NOT NULL,

)

-- assets devices on site
CREATE TABLE devices(
    plant_id INTEGER NOT NULL,
    combiners INTEGER NOT NULL,
    inverters INTEGER NOT NULL,
    primary_meters INTEGER NOT NULL,
    sensors INTEGER NOT NULL,
    weather_stations INTEGER NOT NULL,
    PRIMARY KEY (plant_id),
    CONSTRAINT fkey_plant FOREIGN KEY (plant_id) REFERENCES assets(id)
)

-- asset devices metadata
CREATE TABLE asset_data(
    asset_id INTEGER NOT NULL,
)

