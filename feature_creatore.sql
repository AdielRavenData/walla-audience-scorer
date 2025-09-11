CREATE OR REPLACE TABLE `wallabi-169712.UsersClustering.august_feature` AS
WITH user_events AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS event_date,
    event_timestamp AS time_stamp,
    event_name,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'userUniqueId') AS user_unique_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'vertical_name') AS vertical_name,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'category_id') AS category_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'CategoryName') AS CategoryName,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title') AS page_title,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'item_title') AS item_title,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'tags') AS tags,
    geo.city AS city,
    geo.country AS country,
    device.category AS device_category,
    device.mobile_brand_name AS device_brand,
    device.language AS device_language,
    traffic_source.name AS source_name,
    traffic_source.source AS source,
    EXTRACT(DAYOFWEEK FROM PARSE_DATE('%Y%m%d', event_date)) AS day_of_week,
    EXTRACT(HOUR FROM TIMESTAMP_MICROS(event_timestamp)) AS hour_of_day
  FROM `wallabi-169712.analytics_341158348.events_fresh_*`
  WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', '2025-08-01')
  AND _TABLE_SUFFIX <= FORMAT_DATE('%Y%m%d', '2025-08-31')
    -- WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', '2025-08-01')
)
SELECT DISTINCT
  event_date,
  time_stamp,
  day_of_week,
  hour_of_day,
  event_name,
  user_unique_id,
  CASE
    WHEN city IN ('Acre','Afula','Arraba',"Beit She'an",'Hatzor HaGlilit','Iksal','Kafr Kanna',
      'Kafr Qara','Kfar Tavor','Kfar Vradim',"Ma'ale Iron",'Maghar','Majd al-Krum',
      'Migdal','Nahariyya','Nahf','Qatsrin','Qiryat Shemona','Reineh','Shefa-Amr',
      'Tamra','Tiberias','Umm al-Fahm','Karmiel','Migdal HaEmek','Nof HaGalil',
      'Safed','Nazareth',"Ma'alot-Tarshiha",'Sakhnin','Kafr Manda') THEN 'צפון'
    WHEN city IN ('Atlit','Daliyat al-Karmel','Haifa','Harish','Kfar Bialik','Nesher',
      'Tirat Carmel',"Yokne'am Illit","Zikhron Ya'akov",'Kiryat Ata',
      'Kiryat Bialik',"Kiryat Tiv'on",'Kiryat Motzkin',
      'Kiryat Yam') THEN 'חיפה והקריות'
    WHEN city IN ('Baqa al-Gharbiyye',"Beit Yitzhak-Sha'ar Hefer","Binyamina-Giv'at Ada",
      'Caesarea','Even Yehuda','Hadera','Herzliya','Jatt','Kadima Zoran',
      'Kefar Sava','Kfar Yona',"Kokhav Ya'ir Tzur Yigal",'Netanya',
      'Or Akiva','Pardes Hanna-Karkur',"Ra'anana",'Tayibe','Tel Mond','Tira',
      'Zemer','Hod Hasharon','Qalansawe') THEN 'השרון'
    WHEN city IN ('Bat Yam','Bnei Brak',"Giv'atayim",'Holon','Magshimim','Petah Tikva',
      'Ramat Gan','Ramat Hasharon','Savyon','Tel Aviv-Yafo','Yehud-Monosson',
      'Kiryat Ono','Or Yehuda') THEN 'גוש דן'
    WHEN city IN ("Be'er Ya''akov",'Bet Shemesh','Bitzaron','Gedera','Gan Yavne',
      'Kiryat Ekron','Lod','Mazkeret Batya',"Modi'in-Maccabim-Re'ut",'Nehalim',
      'Nes Ziona','Ramla','Rehovot','Rishon LeZion','Rosh Haayin','Shoham',
      'Yavne','Ness Ziona',"Be'er Ya'akov",'Kfar Qasim') THEN 'מרכז/שפלה'
    WHEN city IN ('Arad','Ashdod','Ashkelon',"Be'er Sheva",'Dimona','Eilat',
      'Kiryat Gat','Kiryat Malakhi','Meitar','Mitzpe Ramon','Omer',
      'Rahat','Sderot','Tel Sheva','Yeruham','Netivot','Ofakim') THEN 'דרום'
    WHEN city IN ('Jerusalem','Mevaseret Zion') THEN 'ירושלים'
    WHEN city IS NULL THEN 'לא ידוע'
    ELSE 'אחר'
  END AS region,
  city,
  country,
  CASE
    WHEN city IN ('Bnei Brak') THEN 'חרדי־רוב'
    WHEN city IN ('Rahat','Tel Sheva','Hura','Lakiya','Kuseife','Segev Shalom') THEN 'בדואי'
    WHEN city IN ('Daliyat al-Karmel','Maghar','Beit Jann','Yarka',"Peki'in",'Hurfeish') THEN 'דרוזי'
    WHEN city IN (
      'Arraba','Sakhnin','Tamra','Tayibe','Tira','Qalansawe','Baqa al-Gharbiyye',
      'Kafr Qara','Kafr Kanna','Kafr Manda','Jatt','Shefa-Amr','Reineh','Nazareth',
      'Majd al-Krum','Iksal','Umm al-Fahm'
    ) THEN 'ערבי־רוב'
    WHEN city IN (
      'Jerusalem','Haifa','Acre','Ramla','Lod',"Ma'alot-Tarshiha"
    ) THEN 'מעורב'
    WHEN country = 'Israel' then 'יהודי־רוב'
    ELSE 'מחוץ לישראל'
  END AS sector,
  device_category,
  device_brand,
  device_language,
  source_name,
  source,
  page_location,
  vertical_name,
  page_title,
  item_title,
  CategoryName,
  category_id,
  tags
FROM user_events
WHERE user_unique_id IS NOT NULL
  AND event_name = "page_view"
  AND vertical_name not in ('וואלה','חדשות')
  AND CategoryName is not null;