import styles from "./styles.module.css";
import { latestReleases } from "../../posts";

const Version = ({ version }: { version: string }) => {
  return <div className={styles.version}>{`v${version}`}</div>;
};

const LatestRelease = () => {
  const { path, version } = latestReleases[0];
  return (
    <a className={styles.a} href={path}>
      <div className={styles.container}>
        <Version version={version} />
        See the latest release
        <img src="img/arrow.svg" alt="" />
      </div>
    </a>
  );
};

export default LatestRelease;
