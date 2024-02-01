import React from "react";
import Layout from "@theme/Layout";
import companies from "../data/companies.json";
import styles from "./powered-by.module.css";
import { H2 } from "../components/Header";

function nameToSrc(name: string) {
  return `img/companies/${name.toLowerCase().replace(/ /g, "-")}.svg`;
}

type CompanyTileProps = {
  name: string;
  url: string;
};

function CompanyTile({ name, url }: CompanyTileProps): React.ReactElement {
  return (
    <a href={url} target="_blank">
      <div className={styles.logoContainer}>
        <img
          className={styles.logo}
          src={nameToSrc(name)}
          alt={name}
          title={name}
        />
      </div>
    </a>
  );
}

export default function Companies(): React.ReactElement {
  return (
    <Layout
      wrapperClassName={styles.pageContainer}
      title="Powered By MLflow"
      description="Powered By MLflow"
    >
      <H2>Powered by MLflow</H2>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          maxWidth: "60rem",
          margin: "auto",
        }}
      >
        <div className={styles.text}>
          MLflow is trusted by organizations of all sizes to power their AI and
          machine learning workflows. Below are some of the organizations using
          and contributing to MLflow. To add your organization here, email our
          user list at{" "}
          <a href="https://groups.google.com/g/mlflow-users">
            mlflow-users@googlegroups.com
          </a>
          !
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          {Object.keys(companies).map((name) => {
            return <CompanyTile name={name} url={companies[name]} />;
          })}
        </div>
      </div>
    </Layout>
  );
}
