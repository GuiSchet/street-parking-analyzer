import { Line, Text } from 'react-konva';
import { useState } from 'react';
import PropTypes from 'prop-types';

const ParkingSpace = ({ space }) => {
  const [hovered, setHovered] = useState(false);

  const getColor = (status) => {
    switch (status) {
      case 'available': return '#10b981'; // green
      case 'occupied': return '#ef4444';   // red
      case 'uncertain': return '#f59e0b'; // yellow
      default: return '#6b7280';          // gray
    }
  };

  if (!space.polygon || space.polygon.length === 0) {
    return null;
  }

  const polygon = space.polygon;
  const flatPoints = polygon.flat();

  return (
    <>
      <Line
        points={flatPoints}
        fill={getColor(space.status)}
        opacity={hovered ? 0.8 : 0.6}
        closed={true}
        stroke="#1f2937"
        strokeWidth={2}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      />
      {hovered && (
        <Text
          x={polygon[0][0]}
          y={polygon[0][1] - 20}
          text={`${space.space_id} (${(space.confidence * 100).toFixed(0)}%)`}
          fontSize={12}
          fill="#ffffff"
          padding={4}
          cornerRadius={4}
        />
      )}
    </>
  );
};

ParkingSpace.propTypes = {
  space: PropTypes.shape({
    space_id: PropTypes.string.isRequired,
    status: PropTypes.string.isRequired,
    confidence: PropTypes.number.isRequired,
    polygon: PropTypes.arrayOf(
      PropTypes.arrayOf(PropTypes.number)
    ).isRequired,
  }).isRequired,
};

export default ParkingSpace;
