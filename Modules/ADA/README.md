<p align="center">
    A <a href="https://nodejs.org/en/">Node.js</a> based, dockerized annotation container.
</p>


# ADA - a tool for Automatic Data Annotation

 Data ecosystems have emerged as versatile platforms for managing and analyzing data from diverse sources, facilitating integration, collaboration and governance across organizations and systems. Annotated data are crucial for efficient and effective large-scale data ecosystems. However, there is a lack of full-fledged automatic annotation approaches for data ecosystems, with manual annotation by experts being the current requirement. Addressing specific annotation requirements of data ecosystems, we introduce ADA, an approach for automatic data annotation. ADA applies a semantic representation model called Data Product Description Object (DPDO) in JSON-LD and combines state-of-the-art models for metadata embeddings within an annotation pipeline. The approach extends technical metadata by essential concepts for data ecosystems, such as data provenance, quality, and accessibility. The effectiveness of ADA was evaluated using competency questions on data sets from diverse domains within the GAIA-X data ecosystem. 

## Semantic Representation Model

Based on [41,42,43,44,45,46] (see section References) we developed the Data Product Description Object (DPDO) serving as basis for our approach. The DPDO contains the following main entities and builds up on existign standard data vocabularies and ontologies, such as [schema.org](https://schema.org/), [data quality vocabulary](https://www.w3.org/TR/vocab-dqv/), [open vocab](https://vocab.org/open/), [data catalog vocabulary](https://www.w3.org/TR/vocab-dcat-3/), [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) and [GAIA-X core](https://gaia-x.gitlab.io/gaia-x-community/gaia-x-self-descriptions/core/core.html) and [participant ontology](https://gaia-x.gitlab.io/technical-committee/service-characteristics/widoco/participant/participant.html)

<img width="1391" alt="DPDOFull" src="https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/6a87d6c4-6444-4a8b-83de-3897ee454e12">

## Evaluation

### Competency Questions (CQ)
Based on several related works (see section References), we identified the following questions to be relevant to participants of data ecosystems.

<img width="994" alt="Screenshot 2023-07-20 at 00 58 15" src="https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/20d323cd-c52c-44d8-8f72-1da20b3f5d6b">

### Data sets

The following open data sets matching the [GAIA-X ecosystem domains](https://gaia-x-hub.de/domaenen/) were used for the evaluation by these competency questions as input to populate our knowledge graph, which was then queried using cypher queries.

[Agriculture](https://www.kaggle.com/datasets/srinivas1/agricuture-crops-production-in-india) <br>
[Energy](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption) <br>
[Construction](https://www.kaggle.com/datasets/vinayakshanawad/cement-manufacturing-concrete-dataset) <br>
[Health-Care](https://www.kaggle.com/datasets/bsridatta/covid-19-italy-updated-regularly) <br>
[Industry 4.0](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot?select=Readme.txt) <br>
[Culture](https://www.kaggle.com/datasets/insiyeah/musicfeatures) <br>
[Mobility](https://www.kaggle.com/datasets/aiswaryaramachandran/google-mobility-data) <br>
[Public Sector](https://www.kaggle.com/datasets/new-york-state/nys-salary-information-for-the-public-sector) <br>
[Education](https://www.kaggle.com/datasets/rajanand/education-in-india) <br>
[Finances](https://www.kaggle.com/datasets/gauravduttakiit/medical-insurance-cost) <br>
[Geo Data](https://www.kaggle.com/datasets/eidanch/counties-geographic-coordinates) <br>
[Smart Cities](https://www.kaggle.com/datasets/magdamonteiro/smart-cities-index-datasets) <br>
[Smart Living](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) <br>

### Exemplary Annotation
Exemplary annotation for the Agriculture data set generated by our demonstrator.
<img width="1411" alt="Agri1" src="https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/1c103f5d-2f48-4b54-bac3-447e0b67d112">
<img width="1414" alt="Agri2" src="https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/9775bc56-cef1-4989-8bc0-e1fd1b3b571f">

### Screencast
A demonstration of the service can be viewed [here](https://www.youtube.com/watch?v=2af0_IButIA&t=1s)

### Cypher Queries
After populating the knowledge graph, we answered all competency questions by using Cypher queries. 

Answer to CQ (1) "Which topic/theme/context is the focus of the dataset?", (2) "What’s the title of the offered data set?":
![theme](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/4ba2b473-7992-455a-8f40-7bf140bcc071)

Answer to CQ (3) "Which content does the data set include in terms of columns?":
![columns](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/bc512059-704f-4a5f-8920-f92bec6fec60)

Answer to CQ (4) "What data types are used?", (5) "In which format/ data structure are the data available?", (6)
"What’s the size of the dataset?":

![dataType](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/b91b1593-3ac8-4a8b-8529-bb15521a1b59)

Answer to CQ (7) "How is the price/value of the data product composed?", (8) "Are contractual transactions described?", (9) "Are there restrictions in usage or accessibility e.g. license, permissions and sanctions?":
![contract](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/a4d4a168-ae33-4ccf-b62d-2c035d803661)

Answer to CQ (10) "What’s the version oft the data set?":
![version](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/a35b201f-61ed-435b-943d-56fdb5901ceb)

Answer to CQ (11) "How often is the dataset updated?":
![Screenshot 2023-07-19 at 01 57 10](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/9ca63ef0-f754-4efb-96db-1548671ec5a7)

Answer to CQ (12) "Are the data up to date and still available?":
![availability](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/cd8d652a-7819-4dcd-9902-dba352b45fd3)

Answer to CQ (13) "What’s the provenance of the dataset?":
![source](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/5dd0d2e1-bcb5-4d23-94f4-e7afa20559a6)

Answer to CQ (14) "Are the data of high quality?":
![Quality](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/820c48d8-6311-48f3-b6fd-bf4e6ed87c00)

Answer to CQ (15) "What is the level of data accuracy?":
![Accuracy](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/3dce5f54-a53b-416e-82b6-8cedfaa15ade)

Answer to CQ (16) "Is there a contact point for technical support?":
![technicalContact](https://github.com/NaGd01/AutomaticDataAnnotation/assets/65232571/5479d930-35f5-42c2-abe7-c9d2b7aa64f4)


## References

### References used to derive Competency Questions

1.	Fassnacht, M., Benz, C., Heinz, D., Leimstoll, J., & Satzger, G. (2023).  Barriers to data sharing among private sector organizations. Proceedings of the 56th Hawaii International Conference on System Sciences Hawaii International Conference on System Sciences (HICSS) 
2.	Van de Wetering, R., & Versendaal, J. (2018). How a flexible collaboration infrastructure impacts healthcare information exchange. BLED 2018 Proceedings, 12.
3.	Stein, H., Rix, C., Effertz, A., Bergau, S., & Maass, W. (2022). Data Sharing in the German Food Industry-Empirical Insights. AMCIS 2022 Proceedings, 1
4.	Abbas, A. E., Agahari, W., Van de Ven, M., Zuiderwijk, A., & De Reuver, M. (2021). Business data sharing through data marketplaces: A systematic literature review. Journal of Theoretical and Applied Electronic Commerce Research, 16(7), 3321-3339.
5.	Parvinen, P. (2020). Advancing data monetization and the creation of data-based business models. Communications of the association for information systems, 47(1), 2.
6.	Gelhaar, J., Gürpinar, T., Henke, M., & Otto, B. (2021). Towards a taxonomy of incentive mechanisms for data sharing in data ecosystems. In PACIS (p. 121).
7.	Gelhaar, J., & Otto, B. (2020). Challenges in the Emergence of Data Ecosystems. PACIS 2020 Proceedings.
8.	Pant, V., & Yu, E. (2018). Getting to win-win in industrial collaboration under coopetition: A strategic modeling approach. Lecture Notes in Business Information Processing, 330, 47–66.
9.	Bastiaansen, H. J. M., Kollenstart, M., Dalmolen, S., & van Engers, T. M. (2020). User-centric network-model for data control with interoperable legal data sharing artefacts: Improved data sovereignty, trust and security for enhanced adoption in interorganizational and supply chain is applications. In 24th Pacific Asia Conference on Information Systems: Information Systems (IS) for the Future, PACIS 2020.
10.	Cichy, P., Salge, T. O., & Kohli, R. (2021). PRIVACY CONCERNS AND DATA SHARING IN THE INTERNET OF THINGS: MIXED METHODS EVIDENCE FROM CONNECTED CARS. MIS Quarterly, 45(4).
11.	Maass, W. (2022). Contract-based Data-Driven Decision Making in Federated Data Ecosystems. Proceedings of the 55th Hawaii International Conference on System Sciences Hawaii International Conference on System Sciences (HICSS) 
12.	Kajüter, P., Arlinghaus, T., Kus, K., & Teuteberg, F. (2022). Analysis of Barriers to Digital Linking among Healthcare Stakeholders. International Conference on Wirtschaftsinformatik 2022 Proceedings.
13.	Van den Broek, T., & van Veenstra, A. F. (2015). Modes of governance in inter-organizational data collaborations. ECIS 2015 Proceedings
14.	Hasan, M. R., & Legner, C. (2023). UNDERSTANDING DATA PRODUCTS: MOTIVATIONS, DEFINITION, AND CATEGORIES. Thirty-first European Conference on Information Systems (ECIS 2023)
15.	Allemang, D., & Teegarden, B. (2017). A global data ecosystem for agriculture and food. F1000Research, 6(1844), 1844.
16.	Spiekermann, M., Tebernum, D., Wenzel, S., & Otto, B. (2018, March). A metadata model for data goods. In Multikonferenz Wirtschaftsinformatik (Vol. 2018, pp. 326-337).
17.	Ehrlinger, L., Schrott, J., Melichar, M., Kirchmayr, N., & Wöß, W. (2021). Data catalogs: a systematic literature review and guidelines to implementation. In Database and Expert Systems Applications-DEXA 2021 Workshops, Proceedings 32 (pp. 148-158). Springer International Publishing
18.	Pesce, V., Maru, A., Archer, P., Malapela, T., & Keizer, J. (2015). Setting up a global linked data catalog of datasets for agriculture. In Metadata and Semantics Research: 9th Research Conference, MTSR 2015, Manchester, UK, September 9-11, 2015, Proceedings 9 (pp. 357-368). Springer International Publishing
19.	Nogueras-Iso, J., Lacasta, J., Ureña-Cámara, M. A., & Ariza-López, F. J. (2021). Quality of metadata in open data portals. IEEE Access, 9, 60364-60382
20.	Neumaier, S., Umbrich, J., & Polleres, A. (2016). Automated quality assessment of metadata across open data portals. Journal of Data and Information Quality (JDIQ), 8(1), 1-29.
21.	Kline, K., McDowell, D., Dorsey, D., & Gordon, M. (2022). Documenting Data Sources and Metadata in a Data Dictionary. In Pro Database Migration to Azure: Data Modernization for the Enterprise (pp. 241-262). Berkeley, CA: Apress.
22.	Solmaz, G., Cirillo, F., Fürst, J., Jacobs, T., Bauer, M., Kovacs, E., ... & Sánchez, L. (2022, December). Enabling data spaces: Existing developments and challenges. In Proceedings of the 1st International Workshop on Data Economy (pp. 42-48).
23.	Nikiforova, A., & McBride, K. (2021). Open government data portal usability: A user-centred usability analysis of 41 open government data portals. Telematics and Informatics, 58, 101539.
24.	Quarati, A., De Martino, M., & Rosim, S. (2021). Geospatial open data usage and metadata quality. ISPRS international journal of geo-information, 10(1), 30.
25.	Janev, V., Vidal, M. E., Pujić, D., Popadić, D., Iglesias, E., Sakor, A., & Čampa, A. (2022). Responsible knowledge management in energy data ecosystems. Energies, 15(11), 3973.
26.	Reiff, S. B., Schroeder, A. J., Kırlı, K., Cosolo, A., Bakker, C., Mercado, L., ... & Park, P. J. (2022). The 4D Nucleome Data Portal as a resource for searching and visualizing curated nucleomics data. Nature communications, 13(1), 2365.
27.	Demchenko, Y., De Laat, C., & Membrey, P. (2014, May). Defining architecture components of the Big Data Ecosystem. In 2014 International conference on collaboration technologies and systems (CTS) (pp. 104-112). IEEE.
28.	Koleti, A., Terryn, R., Stathias, V., Chung, C., Cooper, D. J., Turner, J. P., ... & Schürer, S. C. (2018). Data Portal for the Library of Integrated Network-based Cellular Signatures (LINCS) program: integrated access to diverse large-scale cellular perturbation response data. Nucleic acids research, 46(D1), D558-D566.
29.	Wamhof, T., Bernardi, A., Martini, D., Leinberger, M., Sinha, A., Tapken, H., ... & Graf, H. (2023). Metadata Management and Asset Exchange in the Agricultural Data Ecosystem of the Project Agri-Gaia. Datenbank-Spektrum, 1-9.
30.	Immonen, A., & Kalaoja, J. (2019). Requirements of an energy data ecosystem. IEEE access, 7, 111692-111708.
31.	Drees, H., Kubitza, D. O., Lipp, J., Pretzsch, S., & Langdon, C. S. (2021, October). Mobility data space–first implementation and business opportunities. In Proceedings of the 27th ITS World Congress, Hamburg, Germany (pp. 11-15).
32.	Hayashi, T., & Ohsawa, Y. (2020). Understanding the structural characteristics of data platforms using metadata and a network approach. IEEE Access, 8, 35469-35481.
33.	Curry, E., Curry, E., & Ojo, A. (2020). Enabling knowledge flows in an intelligent systems data ecosystem. Real-time Linked Dataspaces: Enabling Data Ecosystems for Intelligent Systems, 15-43.
34.	Shabani, M. (2022). Will the European Health Data Space change data sharing rules?. Science, 375(6587), 1357-1359.
35.	Shah, S. I. H., Peristeras, V., & Magnisalis, I. (2021). Government big data ecosystem: definitions, types of data, actors, and roles and the impact in public administrations. ACM Journal of Data and Information Quality, 13(2), 1-25.
36.	Courtot, M., Gupta, D., Liyanage, I., Xu, F., & Burdett, T. (2022). BioSamples database: FAIRer samples metadata to accelerate research data management. Nucleic acids research, 50(D1), D1500-D1507.
37.	Jarke, M., Otto, B., & Ram, S. (2019). Data sovereignty and data space ecosystems. Business & Information Systems Engineering, 61, 549-550.
38.	Anwar, M. J., Gill, A. Q., Hussain, F. K., & Imran, M. (2021). Secure big data ecosystem architecture: challenges and solutions. EURASIP Journal on Wireless Communications and Networking, 2021(1), 130.
39.	Shae, Z. Y., & Tsai, J. J. (2021, December). On the design of medical data ecosystem for improving healthcare research and commercial incentive. In 2021 IEEE Third International Conference on Cognitive Machine Intelligence (CogMI) (pp. 124-131). IEEE.
40.	Bader, S., Pullmann, J., Mader, C., Tramp, S., Quix, C., Müller, A. W., ... & Lange, C. (2020, November). The international data spaces information model–an ontology for sovereign exchange of digital content. In International Semantic Web Conference (pp. 176-192). Cham: Springer International Publishing.

### DPDO References <br>
41. Maass, W. (2009). Elektronische Wissensmärkte: Handel von Information und Wissen über digitale Netze. Springer-Verlag <br>
42. Janzen, S., & Maass, W. (2008). Smart product description object (spdo). In Poster Proceedings of the 5th International Conference on Formal Ontology in Information Systems (FOIS) <br>
43. Abramovici, M. (2014). Smart products. CIRP Encyclopedia of Production Engineering, 59, 1-5. <br>
44. Oberweis, A., Pankratius, V., & Stucky, W. (2007). Product lines for digital information products. Information systems, 32(6), 909-939. <br>
45. Hui, K. L., & Chau, P. Y. (2002). Classifying digital products. Communications of the ACM, 45(6), 73-79 <br>
46. Haupert, J. (2013). DOMeMan: Repräsentation, Verwaltung und Nutzung von digitalen Objektgedächtnissen (pp. 1-286). Saarland University. <br>

## Install

### Prerequisits
You need to install <a href="https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04">docker</a> and <a href="">docker-compose</a> for the system to work. **Dont forget to set the docker command to work withou sudo.**

### Build and Run docs and srs container

```sh
docker-compose up --build
```
Executing this command will start two docker container. ReDoc and the application itself. Redoc lets serves a documentation page of the openAPI specification.
![image](https://user-images.githubusercontent.com/37148029/118663163-e19b0880-b7f0-11eb-8e3c-1d9eba33d3ea.png)


### Build and Run single containers using Docker
Build the container by 
```
docker build -t spaicer-example-container . 
```
Run the container 
```
docker run -dp 3001:3001 spaicer-example-container
```

Test the server using curl
```sh
curl -X POST http://<your-ip>:8081/interface/annotate -H "Content-Type: application/json" -d "{\"UploadedData\":data.csv}"
```
returns DPDO file with annotation of data


### Development 

Install dependencies by 
```sh
npm install
```
and start the application by 

```sh
npm run start
```

### Build the docs 
Install dependencies by 
```sh
npm install
```
and run the docs
```sh
npm run docs
```
