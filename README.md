# 🌍 AI-Powered Multi-Pollutant Mapping for Hyperlocal Air Quality Monitoring

> **1st Place & ₹1.5L Grant Winner – Climate Hackathon 2024**  
> *Issued by the U.S. Consulate General Mumbai & Ekonnect Knowledge Foundation*

![Spatial Resolution](https://img.shields.io/badge/Resolution_Enhancement-1000m_→_10m-brightgreen)
![Model Accuracy](https://img.shields.io/badge/Model_Accuracy-88.88%25-blue)
![Domain](https://img.shields.io/badge/Domain-Geospatial_AI_%7C_Remote_Sensing-orange)

---

## 📌 Project Overview

Urban air pollution monitoring often suffers from a critical data scarcity issue: satellite instruments like **Sentinel-5P** provide essential atmospheric pollution data, but at a coarse resolution (~1000m), making it difficult to pinpoint neighborhood-level hotspots. 

This project implements a **domain-agnostic "Super-Resolution" AI pipeline** that fuses coarse-resolution Sentinel-5P atmospheric data with high-resolution **Sentinel-2** surface imagery. Powered by **Convolutional Neural Networks (CNNs)** and statistical correlation techniques, the system downscales pollution maps to **10m spatial resolution**, enabling data-driven, hyperlocal environmental action.

---

## ⚡ Key Features

* **100x Resolution Enhancement:** Successfully downscales satellite pollution measurements from 1000m grid cells to high-definition 10m maps.
* **Multi-Source Data Fusion:** Integrates Google Earth Engine to ingest, align, and process massive geospatial datasets from Sentinel-5P (TROPOMI) and Sentinel-2 (MSI).
* **High Precision Hotspot Identification:** Achieves an **88.88% accuracy rate** in identifying urban pollution concentration hotspots.
* **Extensible & Domain-Agnostic Architecture:** Built to scale beyond atmospheric pollution — the core downscaling pipeline can be retrained for agricultural crop health monitoring, flood risk mapping, and urban heat island analysis.
* **Real-World Deployment:** Currently being deployed across the 65-acre Somaiya Vidyavihar campus (serving 39,000+ students) under the *“Towards Green Campuses”* initiative.

---

## 🛠️ Tech Stack

* **Core & ML:** Python, TensorFlow, Convolutional Neural Networks (CNNs), NumPy, Pandas
* **Geospatial & Remote Sensing:** Google Earth Engine (GEE) API, Sentinel-5P, Sentinel-2
* **Backend & API:** Flask, REST APIs
* **Frontend Dashboard:** React.js, Leaflet / Mapbox API
