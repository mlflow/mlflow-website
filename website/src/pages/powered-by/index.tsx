import React from "react";
import Layout from "@theme/Layout";
import companies from "../../data/companies.json";
import styles from "./index.module.css";
import { H2 } from "../../components/Header";

type CompanyTileProps = {
  name: string;
  url: string;
  src: string;
};

function CompanyTile({ name, url, src }: CompanyTileProps): React.ReactElement {
  return (
    <a href={url} target="_blank">
      <div className={styles.logoContainer}>
        <img className={styles.logo} src={src} alt={name} title={name} />
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
      <div className={styles.logosSectionOuter}>
        <div className={styles.logosSectionInner}>
          {Object.keys(companies).map((name) => (
            <CompanyTile
              name={name}
              url={companies[name].url}
              src={companies[name].src}
            />
          ))}
        </div>
      </div>
    </Layout>
  );
}
