import React from 'react';
import Layout from '@theme/Layout';
import customers from "../data/customers.json";
import styles from "./customers.module.css";
import { H2 } from '../components/Header';

function nameToSrc(name: string) {
    return `img/companies/${name.toLowerCase().replace(/ /g, "-")}.svg`;
}

type CustomerTileProps = {
    name: string;
    url: string;
}

function CustomerTile({name, url}: CustomerTileProps): React.ReactElement {
    return (
        <a href={url} target='_blank'>
        <div className={styles.logoContainer}>
            <img className={styles.logo} src={nameToSrc(name)} alt={name} />
        </div>
        </a>
    )
}

export default function Customers(): React.ReactElement {
  return (
    <Layout wrapperClassName={styles.pageContainer} title="Customers" description="Customers">
        <H2>Powered by MLflow</H2>
        <div
            style={{
              textAlign: "center",
              fontSize: "1.2rem",
              width: "70%",
              margin: "auto",
              paddingTop: "32px",
              paddingBottom: "32px",
            }}
          >
            MLflow is used by organizations of all sizes, from startups to Fortune 100 companies.
            Here are some of the organizations using and contributing to MLflow:
          </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          flexWrap: 'wrap',
        }}>
        {Object.keys(customers).map((name) => {
            return <CustomerTile name={name} url={customers[name]} />
        })}
      </div>
    </Layout>
  );
}