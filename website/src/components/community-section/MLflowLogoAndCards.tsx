import React, { useEffect, useState } from "react";
import CommunityCard from "./CommunityCard";
import "./MLflowLogoAndCards.css";
import Arrow from "./Arrow";
import ArrowText from "../ArrowText";
import styles from "./styles.module.css";

const MOBILE_LAYOUT_BREAKPOINT = 996;

const MLflowLogo = ({ displaySideDots }: { displaySideDots: boolean }) => {
  return (
    <svg
      className="logo-circle-svg"
      viewBox="0 0 324 324"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* main logo */}
      <circle
        vectorEffect="non-scaling-stroke"
        strokeWidth="2"
        className="lines"
        cx="50%"
        cy="50%"
        r="48%"
        fill="none"
      />
      <g id="grow">
        <g id="mlflow-logo">
          <path
            d={
              "M228.24 66.6196C200.209 46.3229 164.677 39.8923 131.711 49.1363C82.9527 " +
              "63.2033 47.2232 109.424 48.0128 163.481C48.4076 190.208 57.488 214.524 " +
              "72.2931 234.017L107.233 207.893C97.363 195.634 91.4409 179.96 91.2435 " +
              "162.878C90.8487 122.285 122.63 89.1268 162.505 88.5239L161.716 116.256L228.24 66.6196Z"
            }
            fill="#43C9ED"
          />
          <path
            d={
              "M253.909 92.5495C252.725 90.9419 251.541 89.3342 250.356 87.7266L216.798 " +
              "112.846C228.247 125.708 235.354 142.588 235.551 161.277C235.946 201.87 " +
              "204.164 235.028 164.289 235.631L165.079 207.899L97.7654 258.138C136.653 " +
              "284.866 189.557 285.469 229.629 255.526C280.361 217.344 291.415 144.397 253.909 92.5495Z"
            }
            fill="#0194E2"
          />
        </g>
      </g>

      {/* connector dots */}
      {displaySideDots && (
        <g>
          <path
            vectorEffect="non-scaling-stroke"
            strokeLinecap="round"
            strokeWidth="9"
            className="lines"
            d="M6.5 162 H6.5001"
          />
          <path
            vectorEffect="non-scaling-stroke"
            strokeLinecap="round"
            strokeWidth="9"
            className="lines"
            d="M317.5 162 H317.5001"
          />
        </g>
      )}
      <path
        vectorEffect="non-scaling-stroke"
        strokeLinecap="round"
        strokeWidth="9"
        className="lines"
        d="M162 317.5 H162.0001"
      />
    </svg>
  );
};

const LineConnector = ({
  direction,
}: {
  direction: "horizontal" | "vertical";
}) => {
  if (direction === "vertical") {
    return (
      <svg
        height="100%"
        width="100"
        strokeWidth={2}
        className="connector"
        viewBox="0 0 100 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          vectorEffect="non-scaling-stroke"
          className="lines"
          d="M50 0 V 200"
        />
      </svg>
    );
  }

  return (
    <svg
      height="100"
      width="100%"
      strokeWidth={2}
      className="vertical-connector"
      viewBox="0 0 200 100"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        vectorEffect="non-scaling-stroke"
        className="lines"
        d="M0 50 H 200"
      />
    </svg>
  );
};

const GetConnectedCard = ({ isMobile }: { isMobile: boolean }) => {
  const content = (
    <ul
      className="follow-along-list"
      style={{
        listStyleType: "none",
        padding: 0,
      }}
    >
      <li>
        <ArrowText
          text={
            <a
              className={styles.a}
              href="https://join.slack.com/t/mlflow-users/shared_invite/zt-1iffrtbly-UNU8hV03aV8feUeGmqf_uA"
            >
              Join 10,000+ ML practitioners in Slack
            </a>
          }
        />
      </li>
      <li>
        <ArrowText
          text={
            <a
              className={styles.a}
              href="https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md"
            >
              Learn how to contribute
            </a>
          }
        />
      </li>
    </ul>
  );
  return isMobile ? (
    <CommunityCard
      className="community-card-mobile"
      title="Get connected"
      content={content}
      dotPositions={new Set(["top", "bottom"])}
    />
  ) : (
    <CommunityCard
      className="community-card-inline"
      title="Get connected"
      content={content}
      dotPositions={new Set(["right"])}
    />
  );
};

const FollowAlongCard = ({ isMobile }: { isMobile: boolean }) => {
  const content = (
    <ul
      className="follow-along-list"
      style={{
        listStyleType: "none",
        padding: 0,
      }}
    >
      <li>
        <ArrowText
          text={
            <a className={styles.a} href="https://github.com/mlflow">
              View code on GitHub
            </a>
          }
        />
      </li>
      <li>
        <ArrowText
          text={
            <a className={styles.a} href="https://x.com/mlflow">
              Follow us on X (formerly known as Twitter)
            </a>
          }
        />
      </li>
      <li>
        <ArrowText
          text={
            <a
              className={styles.a}
              href="https://www.linkedin.com/company/mlflow-org"
            >
              Follow us on LinkedIn
            </a>
          }
        />
      </li>
    </ul>
  );
  return isMobile ? (
    <CommunityCard
      className="community-card-mobile"
      title="Follow along"
      content={content}
      dotPositions={new Set(["top", "bottom"])}
    />
  ) : (
    <CommunityCard
      className="community-card-inline"
      title="Follow along"
      content={content}
      dotPositions={new Set(["left"])}
    />
  );
};

const SubscribeCard = ({ isMobile }: { isMobile: boolean }) => {
  return (
    <CommunityCard
      className={isMobile ? "community-card-mobile" : "community-card-block"}
      title="Subscribe to our mailing list"
      content={
        <ArrowText
          text={
            <a
              className={styles.a}
              href="https://groups.google.com/g/mlflow-users"
            >
              mlflow-users@googlegroups.com
            </a>
          }
        />
      }
      dotPositions={new Set(["top"])}
    />
  );
};

const FullWidthLayout = () => {
  return (
    <div className="logo-container-full">
      {/* row 1 */}
      <GetConnectedCard isMobile={false} />
      <LineConnector direction="horizontal" />
      <MLflowLogo displaySideDots={true} />
      <LineConnector direction="horizontal" />
      <FollowAlongCard isMobile={false} />
      {/* row 2 */}
      <div
        className="vertical-connector-container"
        style={{ gridColumn: "1/-1" }}
      >
        <LineConnector direction="vertical" />
      </div>
      {/* row 3 */}
      <div style={{ gridColumn: "1/-1" }}>
        <SubscribeCard isMobile={false} />
      </div>
    </div>
  );
};

const MobileLayout = () => {
  return (
    <div className="logo-container-mobile">
      <MLflowLogo displaySideDots={false} />
      <div style={{ height: 32 }}>
        <LineConnector direction="vertical" />
      </div>
      <GetConnectedCard isMobile={true} />
      <div style={{ height: 32 }}>
        <LineConnector direction="vertical" />
      </div>
      <FollowAlongCard isMobile={true} />
      <div style={{ height: 32 }}>
        <LineConnector direction="vertical" />
      </div>
      <SubscribeCard isMobile={true} />
    </div>
  );
};

const MLflowLogoAndCards = () => {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return windowWidth > MOBILE_LAYOUT_BREAKPOINT ? (
    <FullWidthLayout />
  ) : (
    <MobileLayout />
  );
};

export default MLflowLogoAndCards;
